
import pymupdf4llm
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
import glob
import ollama
from tqdm import tqdm
from pathlib import Path 
import logging 
from uuid import uuid4 
from typing import List
import re 
import shutil 


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(name)s - %(filename)s: %(message)s")

logger = logging.getLogger(__name__)
chroma_path = Path.cwd()
chroma_file = "chroma_db"
#delete previous version

chroma_db_path = Path(chroma_path, chroma_file)
if chroma_db_path.exists():
    logger.warning(f"{chroma_db_path} exists - deleting before re-ingest")
    shutil.rmtree(chroma_db_path)


### Data Loading ### 
images_dir = Path(Path.cwd(), "extracted_images")
if not images_dir.exists():
    images_dir.mkdir(parents=True)

pdf_dir = Path(Path.cwd(),"papers_and_books")

gemma_model = "gemma4:e4b" 
documents_to_embed:List[Document] = []

pdf_files = glob.glob(f"{pdf_dir}/*.pdf")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, 
    chunk_overlap=100
)


def get_img_page_and_no(img_path:str | Path):
    if isinstance(img_path, Path):
        name_parts = img_path.name.split('.pdf')
    else:
        name_parts = img_path.split('.pdf')
    source_file = name_parts[0]
    img_page_and_no = name_parts[1].split('-')
    page_no = img_page_and_no[1]
    image_no = img_page_and_no[2].split('.')[0]
    return source_file, page_no, image_no


def split_by_headers(page_text: str, current_headers: dict) -> list[dict]:
    """
    Parse headers from page text, updating current_headers in place.
    current_headers carries state across pages so sections aren't lost
    at page boundaries.
    Returns: [{"h1": str, "h2": str, "h3": str, "text": str}]
    """
    sections = []
    current_lines = []

    def flush():
        text = "\n".join(current_lines).strip()
        if text:
            # Snapshot current header state at flush time
            sections.append({**current_headers, "text": text})
        current_lines.clear()

    for line in page_text.split("\n"):
        # #### **Header text** or #### Header text
        if line.startswith("####"):
            flush()
            current_headers["h3"] = re.sub(r'\*+', '', line[4:]).strip()

        elif line.startswith("###"):
            flush()
            current_headers["h3"] = re.sub(r'\*+', '', line[3:]).strip()

        elif line.startswith("##"):
            flush()
            current_headers["h2"] = re.sub(r'\*+', '', line[2:]).strip()
            current_headers["h3"] = ""

        elif line.startswith("#"):
            flush()
            current_headers["h1"] = re.sub(r'\*+', '', line[1:]).strip()
            current_headers["h2"] = ""
            current_headers["h3"] = ""

        # Bold-only subsection lines like "**1.2.1** **Sum**"
        elif re.match(r'^(\*\*[^*]+\*\*\s*){1,3}$', line.strip()) and len(line.strip()) < 80:
            flush()
            current_headers["h3"] = re.sub(r'\*+', '', line).strip()

        else:
            current_lines.append(line)

    flush()
    return sections


BLOCK_MATH_RE = re.compile(r'(.{0,400})\$\$(.*?)\$\$(.{0,400})', re.DOTALL)

def extract_formula_chunks(
    page_text: str,
    doc_name: str,
    page_num: int,
    page_id: str,
) -> tuple[list[Document], str]:
    """
    Extract block LaTeX equations from page text.
    Returns:
        formula_docs: List of Document objects, one per formula
        cleaned_text: page_text with $$ blocks removed (for normal chunking)
    """
    formula_docs = []

    for i, match in enumerate(BLOCK_MATH_RE.finditer(page_text)):
        context_before = match.group(1).strip()
        raw_formula = match.group(2).strip()
        context_after = match.group(3).strip()

        if not raw_formula:
            continue

        formula_docs.append(Document(
            page_content=(
                f"Context before: {context_before}\n"
                f"Formula: $${raw_formula}$$\n"
                f"Context after: {context_after}"
            ),
            metadata={
                "type": "formula",
                "document": doc_name,
                "page_no": page_num,
                "parent_id": page_id,
                "formula_id": f"{page_id}_f{i}",
                "raw_formula": raw_formula,   # clean copy for Layer 2 code-gen
            }
        ))

    # Remove extracted block equations from text so the splitter doesn't see them
    cleaned_text = BLOCK_MATH_RE.sub(' ', page_text)
    return formula_docs, cleaned_text


def slugify(text: str) -> str:
    """Convert a header string to a clean key fragment."""
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')[:40]

def process_document(
    doc_path: Path,
    text_splitter: RecursiveCharacterTextSplitter,
    image_path: Path,
    documents_to_embed: list[Document],
):
    doc_name = doc_path.stem
    page_chunks = pymupdf4llm.to_markdown(
        doc_path,
        page_chunks=True,
        write_images=True,
        image_format="jpeg",
        image_path=image_path,
    )

    # Header state persists across pages
    current_headers = {"h1": "", "h2": "", "h3": ""}

    for page_data in page_chunks:
        page_num = page_data["metadata"]["page"]
        page_id = f"{doc_name}_p{page_num}"
        page_text = page_data["text"]

        if not page_text.strip():
            continue

        # Skip bibliography pages
        lines = [l for l in page_text.strip().split("\n") if l.strip()]
        citation_lines = sum(1 for l in lines if re.search(r'\(\d{4}\)', l))
        if lines and citation_lines / len(lines) > 0.6:
            logger.info(f"Skipping bibliography page {page_num} in {doc_name}")
            continue

        formula_docs, cleaned_text = extract_formula_chunks(
            page_text, doc_name, page_num, page_id
        )
        documents_to_embed.extend(formula_docs)

        # Pass current_headers in — it gets mutated in place as new headers are found
        sections = split_by_headers(cleaned_text, current_headers)

        for section in sections:
            header_key = slugify(
                section["h2"] or section["h1"] or f"p{page_num}"
            )
            section_id = f"{doc_name}_{header_key}"
            chunks = text_splitter.split_text(section["text"])
            for chunk in chunks:
                if not chunk.strip():
                    continue
                documents_to_embed.append(Document(
                    page_content=chunk,
                    metadata={
                        "type": "text",
                        "document": doc_name,
                        "page_no": page_num,
                        "parent_id": section_id,
                        "page_id": page_id,
                        "chunk_id": str(uuid4()),
                        "section_h1": section["h1"],
                        "section_h2": section["h2"],
                        "section_h3": section["h3"],
                    }
                ))
                logger.info(f'current embedding list size {len(documents_to_embed)}')

def add_images(image_path:Path, documents_to_embed:List[Document]):
    image_files = list(image_path.glob("*.jpeg"))
    for img_path in tqdm(image_files):
        img_path = str(img_path.absolute())
        try:
            # Prompting Gemma to analyze the image content
            response = ollama.chat(
                model=gemma_model,
                messages=[{
                    'role': 'user',
                    'content': 'Describe this scientific image or figure in detail. What is it showing?',
                    'images': [img_path]
                }]
            )
            summary = response['message']['content']
            source_file, page_no, image_no = get_img_page_and_no(img_path)
            # Package into a structured document
            documents_to_embed.append(Document(
                page_content=f"Image Summary: {summary}",
                metadata={
                    "type": "image", 
                    "source_path": img_path,
                    "document": source_file,
                    "page_no": page_no,
                    "image_no": image_no,
                    "parent_id":f"{source_file}_p{page_no}"
                }
            ))
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}")


for document in pdf_dir.glob("*.pdf"):
    process_document(document, text_splitter, images_dir, documents_to_embed)
logger.info(f"found {len(documents_to_embed)} documents for text")
add_images(images_dir, documents_to_embed)
logger.info(f"after adding images {len(documents_to_embed)}")

embeddings = OllamaEmbeddings(model='embeddinggemma:300m')
vector_store = Chroma.from_documents(
    documents=documents_to_embed,
    embedding=embeddings,
    persist_directory=str(chroma_db_path)
)
