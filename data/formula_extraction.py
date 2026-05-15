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

annotation_llm = ChatOllama(model="gemma4:e4b").with_structured_output(EquationAnnotation)

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
    
    # 3. Pull text siblings — chunks with the same parent_id (same page)
    text_siblings = vector_store.get(
        where={
            "$and": [
                {"parent_id": {"$eq": parent_id}},
                {"type": {"$eq": "text"}}
            ]
        },
        include=["documents", "metadatas"]
    )
    
    return {
        "img_path": img_path.name,
        "parent_id": parent_id,
        "latex": cleaned,
        "latex_degenerate": degenerate,
        "image_summary": img_summary[:400],
        "text_siblings": get_text_siblings(img_meta,vector_store)
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

if __name__ == "__main__":
    # Run on your 10 sample images
    sample_images = list(Path("extracted_images").glob("*.jpeg"))[:5]

    for img in sample_images:
        logger.warning(f"processing {img.name}")
        record = get_image_record(img)
        logger.info(f"\n{'='*70}")
        logger.info(f"Image: {record.get('img_path',None)}")
        logger.info(f"Parent ID: {record.get('parent_id',None)}")
        logger.info(f"\nLaTeX: {record.get('latex',None)}")
        logger.info(f"Degenerate: {record.get('latex_degenerate',None)}")
        logger.info(f"\nImage Summary (Layer 1):\n  {record.get('image_summary',None)}")
        logger.warning(f"\nText Siblings ({len(record.get('text_siblings', []))} chunks):")
        for sib in record.get('text_siblings', []):
            logger.info(sib)
        latex = record.get('latex',None)
        text_siblings = record.get("text_siblings",[])
        if latex != None and len(text_siblings) != 0:
            annotated_result = annotate_equation(latex, text_siblings)
            logger.info(f"{annotated_result=}")
        else:
            logger.info(f"no latex or text_sibling found, skipping for img {img.stem}")