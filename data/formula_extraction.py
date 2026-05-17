from pathlib import Path
from langchain.messages import HumanMessage
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from pix2tex.cli import LatexOCR
from PIL import Image
import re
from langchain_ollama import ChatOllama
import logging 
from typing import List, Optional, Literal
from typing_extensions import TypedDict
from datetime import datetime 
from langchain_core.documents import Document 
import json 
import shutil 

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(filename)s: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.propagate = False 

class Variable(TypedDict):
    symbol: str 
    meaning: str 
    unit: Optional[str]

class EquationAnnotation(TypedDict):
    description: str 
    equation_number: Optional[str]
    role_in_derivation: Optional[str]
    variables: List[Variable]
    biomechanical_context: Literal[
        "imu_orientation",
        "imu_state_estimation",
        "rigid_body_kinematics",
        "numerical_integration",
        "math_property",
        "general_math"
    ]
    python_signature_hint: Optional[str] 
MODEL_NAME = "gemma4:e4b"
annotation_llm = ChatOllama(model=MODEL_NAME).with_structured_output(EquationAnnotation)

SEMANTIC_ANNOTATION_PROMPT = """You are annotating a mathematical equation from a research paper.

The equation has been transcribed for you (do not modify it):
LaTeX: {latex}

Context from the same page of the paper (figures referenced as [FIGURE]):
{page_text}

Fill in each field:

- description: one sentence stating what this equation computes or asserts
- equation_number: the number from the LaTeX if present (e.g. "63", "286"), else null
- role_in_derivation: one of "definition", "intermediate_step", "final_result", "property"
- is_implementable: true if this equation defines a computation with clear inputs and
  outputs that can be written as a Python function. False for properties, constraints,
  identities, or equivalences (e.g., orthogonality conditions).
- variables: list of every symbol with its meaning and unit (null if unitless)
- biomechanical_context: pick ONE:
    * imu_orientation       — attitude/quaternion tracking
    * imu_state_estimation  — Kalman filters, ESKF, covariance updates
    * rigid_body_kinematics — rigid body motion, angular velocity, frame transforms
    * numerical_integration — ODE solvers, RK4, Euler methods
    * math_property         — property, constraint, or identity
    * general_math          — generic with no specific application
- python_signature_hint: a function signature like
  "def quaternion_exp(q: np.ndarray) -> np.ndarray" if is_implementable is true, else null.
"""

embeddings = OllamaEmbeddings(model='embeddinggemma:300m')
vector_store = Chroma(
    persist_directory=str(Path.cwd() / "chroma_db"),
    embedding_function=embeddings
)
LAYER2_DB = Path(Path.cwd() , f"chroma_db_layer2_{MODEL_NAME}")
if LAYER2_DB.exists():
    logger.warning(f"previous version of layer2 found. Deleting to refresh")
    shutil.rmtree(LAYER2_DB)

layer2_store = Chroma(
    persist_directory=str(LAYER2_DB),
    embedding_function=embeddings   # reuse the same embedding model as Layer 1
)
model = LatexOCR()


def clean_pix2tex_output(latex: str) -> str:
    latex = re.sub(r'(\s*~\s*){3,}', ' ', latex)
    return latex.strip().rstrip(',').rstrip('.').rstrip()


def is_degenerate(latex: str) -> bool:
    if latex.count('\\scriptstyle') > 5:
        return True
    non_structural = re.sub(r'[\\{}\s]', '', latex)
    if len(latex) > 100 and len(non_structural) / len(latex) < 0.15:
        return True
    return False

def get_text_siblings(img_meta:dict, vector_store:Chroma):
    img_page = int(img_meta['page_no'])
    img_doc = Path(img_meta['document']).stem
    img_doc = img_doc.replace(' ','-')

    logger.info(f"{img_page=}, {img_doc=}")
    results = vector_store.get(where={"$and":[
        # {"document":{"$eq":img_doc}}, 
        {"page_no":{"$eq":img_page}},
        {"type":{"$eq":"text"}}]})
    return list(zip(results["documents"], results["metadatas"]))

def get_image_record(img_path: Path) -> dict:
    """Pull everything Layer 2 would need for one image."""
    # 1. Run pix2tex
    raw_latex = model(Image.open(img_path))
    cleaned = clean_pix2tex_output(raw_latex)
    degenerate = is_degenerate(cleaned)
    
    # 2. Find the matching image doc in Chroma to get parent_id
    img_results = vector_store.get(
        where={"source_path": str(img_path.absolute())},
        include=["metadatas", "documents"]
    )
    
    if not img_results["ids"]:
        return {"error": "No matching image doc in Chroma", "img_path": str(img_path)}
    
    img_meta = img_results["metadatas"][0]
    img_summary = img_results["documents"][0]
    parent_id = img_meta.get("parent_id")
    img_page = int(img_meta.get('page_no',None))
    if img_page == None:
        logger.error(f"image page not found {img_meta=}")
        
    
    # 3. Pull text siblings — chunks with the same parent_id (same page)
    text_siblings = vector_store.get(
        where={
            "$and": [
                {"page_no": {"$eq": img_page}},
                {"type": {"$eq": "text"}}
            ]
        },
        include=["documents", "metadatas"]
    )
    
    text_section = vector_store.get(where={"$and":[{"section_h1":{"$eq":text_siblings['metadatas'][0]['section_h1']}},
                                                  {"section_h2":{"$eq":text_siblings['metadatas'][0]['section_h2']}},
                                                  {"section_h3":{'$eq':text_siblings['metadatas'][0]['section_h3']}},
                                                  {"type": {"$eq": "text"}},
                                                  {"page_no": {"$eq": img_page-1}},
                                                  {"page_no": {"$eq": img_page}},
                                                  {"page_no": {"$eq": img_page+1}},
                                                  ]},
                                                  include=['documents','metadatas'])
    text_siblings = list(zip(text_siblings['documents'], text_siblings['metadatas']))
    logging.info("*"*10)
    logging.info(f"{text_section=}")
    logging.info("*"*10)
    return {
        "img_path": img_path.name,
        "parent_id": parent_id,
        "latex": cleaned,
        "latex_degenerate": degenerate,
        "image_summary": img_summary[:400],
        "text_siblings": text_siblings
    }

def clean_text_for_annotation(text: str) -> str:
    # Remove pymupdf4llm image embeds
    text = re.sub(r'!\[\]\(extracted_images/[^)]+\)', '[FIGURE]', text)
    # Collapse the multiple blank lines that result
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

def annotate_equation(latex: str, text_siblings: list) -> EquationAnnotation:
    combined_text = "\n\n".join(
        clean_text_for_annotation(doc)
        for doc, _ in text_siblings
    )[:3000]

    prompt = SEMANTIC_ANNOTATION_PROMPT.format(
        latex=latex,
        page_text=combined_text,
    )

    return annotation_llm.invoke([HumanMessage(content=prompt)])

def get_context_for_image(img_metadata: dict, vector_store: Chroma) -> dict:
    """
    Retrieve text siblings. If sparse, also get sibling images on the same page.
    Returns a context dict with all available material.
    """
    page_no = img_metadata['page_no']
    
    # Primary retrieval: text on same page
    text_results = vector_store.get(
        where={
            "$and": [
                {"page_no": {"$eq": page_no}},
                {"type": {"$eq": "text"}}
            ]
        },
        include=["documents", "metadatas"]
    )
    
    text_siblings = [
        (doc, meta) for doc, meta in 
        zip(text_results["documents"], text_results["metadatas"])
    ] if text_results["ids"] else []
    
    # If sparse text, also retrieve other images on same page for context
    sibling_images = []
    if len(text_siblings) < 2:  # threshold: fewer than 2 text chunks
        image_results = vector_store.get(
            where={
                "$and": [
                    {"page_no": {"$eq": page_no}},
                    {"type": {"$eq": "image"}},
                    {"source_path": {"$ne": img_metadata.get("source_path")}}  # exclude self
                ]
            },
            include=["documents", "metadatas"],
            limit=5
        )
        sibling_images = [
            (doc, meta) for doc, meta in 
            zip(image_results["documents"], image_results["metadatas"])
        ] if image_results["ids"] else []
    
    return {
        "text_siblings": text_siblings,
        "image_siblings": sibling_images,
        "page_no": page_no,
    }

def annotate_equation_with_fallback(latex: str, img_metadata: dict, vector_store: Chroma, annotation_llm) -> EquationAnnotation:
    """
    Annotate an equation. On text-sparse pages, supplement with image context.
    """
    context = get_context_for_image(img_metadata, vector_store)
    
    text_content = "\n\n".join(
        clean_text_for_annotation(doc)
        for doc, _ in context["text_siblings"]
    )[:2000]
    
    # If we got sibling images, include their summaries as context
    image_context = ""
    if context["image_siblings"]:
        image_summaries = "\n\n".join(
            f"[Related figure on same page]: {doc[:500]}"
            for doc, _ in context["image_siblings"]
        )
        image_context = "\n\n" + image_summaries
    
    combined_context = text_content + image_context
    
    prompt = SEMANTIC_ANNOTATION_PROMPT.format(
        latex=latex,
        page_text=combined_context[:3000] if combined_context else "[No text context available on this page]",
    )
    
    return annotation_llm.invoke([HumanMessage(content=prompt)])

def assess_latex_quality(latex: str) -> str:
    """Rough quality check for pix2tex output."""
    if latex.count("{") != latex.count("}"):
        return "low"
    if "\\!\\!" in latex or "\\mathrm{{" in latex:  # spacing hacks suggest OCR confusion
        return "low"
    if len(re.findall(r'\\[a-zA-Z]+', latex)) > 20:  # too many commands
        return "low"
    return "ok"

def store_annotation(annotation: EquationAnnotation,latex: str,latex_quality: str,img_metadata: dict,layer2_store: Chroma,):
    # Build the searchable text (this is what gets embedded)
    variable_text = "; ".join(
        f"{v['symbol']}: {v['meaning']}"
        for v in annotation.get("variables", [])
    )
    
    searchable_text = (
        f"{annotation['description']}\n"
        f"Context: {annotation['biomechanical_context']}\n"
        f"Variables: {variable_text}"
    )
    
    layer2_store.add_documents([
        Document(
            page_content=searchable_text,
            metadata={
                "type": "annotated_equation",
                "raw_latex": latex,
                "latex_quality": latex_quality,
                "equation_number": annotation.get("equation_number") or "",
                "is_implementable": annotation.get("is_implementable", False),
                "python_signature_hint": annotation.get("python_signature_hint") or "",
                "biomechanical_context": annotation["biomechanical_context"],
                "role_in_derivation": annotation.get("role_in_derivation") or "",
                "variables_json": json.dumps(annotation.get("variables", [])),
                "source_document": img_metadata["document"],
                "source_page": img_metadata["page_no"],
                "source_image_path": img_metadata["source_path"],
                # Cross-reference back to Layer 1
                "layer1_parent_id": img_metadata.get("parent_id", ""),
            }
        )
    ])

def run_on_image(img_path:Path|str):
    """
    Run all steps one by one on a single image
    """
    img_record = get_image_record(img_path)
    img_latex = img_record.get("latex",None)
    img_latex_quality = assess_latex_quality(img_latex)
    img_metadata = vector_store.get(where={"source_path":{"$eq":str(img_path.absolute())}},include=["metadatas"])
    img_metadata = img_metadata['metadatas'][0]
    if img_latex == None:
        raise Exception(f"Unable to annotate no latex info found for image {img_path}")
    annotation = annotate_equation_with_fallback(img_latex,img_metadata,vector_store, annotation_llm)
    store_annotation(annotation, img_latex, img_latex_quality, img_metadata, layer2_store)
    return annotation




if __name__ == "__main__":
    sample_images = list(Path("extracted_images").glob("*.jpeg"))

    for img in sample_images:
        try:
            start = datetime.now()
            annotation = run_on_image(img)
            logger.info(f"{img.stem} took {datetime.now() - start}")
            logger.info(annotation)
        except Exception as ex:
            logger.error(ex)

    


