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
from tqdm import tqdm 

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
    is_implementable: bool 
    variables: List[Variable]
    biomechanical_context: Literal[
        "impact_mechanics",
        "energy_calculations", 
        "motion_classification",
        "feature_extraction",
        "imu_kinematics",
        "math_property",
        "general_math",
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
    * impact_mechanics       — collisions, momentum exchange, force/impulse estimation
    * energy_calculations    — kinetic/potential energy, work-energy theorem
    * motion_classification  — ML-based activity/gesture recognition  
    * feature_extraction     — time-series features from sensor data
    * imu_kinematics         — orientation, integration, angular velocity
    * math_property          — property, constraint, or identity (not implementable)
    * general_math           — generic with no specific application
- python_signature_hint: a function signature like
  "def quaternion_exp(q: np.ndarray) -> np.ndarray" if is_implementable is true, else null.
"""

embeddings = OllamaEmbeddings(model='embeddinggemma:300m')
vector_store = Chroma(
    persist_directory=str(Path.cwd(),"data","chroma_db"),
    embedding_function=embeddings
)
LAYER2_DB = Path(Path.cwd() , "data", f"chroma_db_layer2_{MODEL_NAME}")
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
    raw_latex = model(Image.open(img_path))
    cleaned = clean_pix2tex_output(raw_latex)
    degenerate = is_degenerate(cleaned)
    
    img_results = vector_store.get(
        where={"source_path": str(img_path.absolute())},
        include=["metadatas", "documents"]
    )
    
    if not img_results["ids"]:
        return {"error": "No matching image doc in Chroma", "img_path": str(img_path)}
    
    img_meta = img_results["metadatas"][0]
    img_summary = img_results["documents"][0]
    img_page = int(img_meta["page_no"])
    
    text_results = vector_store.get(
        where={
            "$and": [
                {"page_no": {"$eq": img_page}},
                {"type": {"$eq": "text"}},
            ]
        },
        include=["documents", "metadatas"]
    )
    text_siblings = list(zip(text_results["documents"], text_results["metadatas"]))
    
    return {
        "img_path": img_path.name,
        "parent_id": img_meta.get("parent_id"),
        "latex": cleaned,
        "latex_degenerate": degenerate,
        "image_summary": img_summary[:400],
        "text_siblings": text_siblings,
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

def run_on_image(img_path: Path | str):
    img_record = get_image_record(img_path)
    img_latex = img_record.get("latex")
    
    if img_latex is None or img_record.get("latex_degenerate"):
        logger.info(f"Skipping {img_path.name}: degenerate or missing LaTeX")
        return None
    
    img_latex_quality = assess_latex_quality(img_latex)
    if img_latex_quality == "low":
        logger.info(f"Skipping {img_path.name}: low quality LaTeX")
        return None
    
    img_metadata_results = vector_store.get(
        where={"source_path": {"$eq": str(img_path.absolute())}},
        include=["metadatas"]
    )
    img_metadata = img_metadata_results["metadatas"][0]
    
    annotation = annotate_equation_with_fallback(
        img_latex, img_metadata, vector_store, annotation_llm
    )
    store_annotation(annotation, img_latex, img_latex_quality, img_metadata, layer2_store)
    return annotation




if __name__ == "__main__":
    sample_images = list(Path("data","extracted_images").glob("*.jpeg"))

    for img in tqdm(sample_images):
        try:
            start = datetime.now()
            annotation = run_on_image(img)
        except Exception as ex:
            logger.error(ex)

    


