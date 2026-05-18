from typing import Annotated

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pathlib import Path 
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Annotated, Sequence, TypedDict
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(filename)s: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.propagate = False 

LAYER2_DB = Path(Path.cwd(),"chroma_db_layer2_gemma4:e4b")
embeddings = OllamaEmbeddings(model='embeddinggemma:300m')
layer2_store = Chroma(persist_directory=str(LAYER2_DB), embedding_function=embeddings)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

CODE_GEN_SYSTEM_PROMPT = """<|think|>You are a code generation assistant specializing in 
biomechanics and motion analysis. You write Python implementations of mathematical 
formulas from research papers.

WORKFLOW (you MUST follow this):
1. First, call retrieve_implementable_formulas with a description of what the user needs.
2. Examine the retrieved formulas. Each has: LaTeX, description, variables, suggested signature.
3. If retrieved formulas don't match the request, retrieve again with a refined query 
   before answering. Do NOT generate code from your own knowledge.
4. Write Python using NumPy. Implement the formula EXACTLY as the LaTeX shows.
5. Cite the source equation number and paper in code comments.

CRITICAL RULES:
- If no implementable formula is retrieved, say so. Do not fabricate.
- If the LaTeX looks garbled or ambiguous, flag this in your response.
- Add docstrings stating the source equation.
"""


@tool 
def retrieve_implementable_formulas(query: str, k: int = 4) -> str:
    """Retrieve formulas that can be implemented as Python functions, relevant to the query.
    
    Args:
        query: Natural description of what the code needs to compute
        k: Number of formulas to retrieve (default 4)
    
    Returns:
        Retrieved formulas with their LaTeX, descriptions, variables, and signature hints.
    """
    retriever = layer2_store.as_retriever(
        search_kwargs={
            "k": k,
            "filter": {
                "$and": [
                    {"type": {"$eq": "annotated_equation"}},
                    {"is_implementable": {"$eq": True}},
                    {"latex_quality": {"$ne": "low"}},  # skip garbled OCR
                ]
            }
        }
    )
    docs = retriever.invoke(query)
    
    results = []
    for doc in docs:
        m = doc.metadata
        results.append(
            f"=== Equation ({m.get('equation_number', '?')}) ===\n"
            f"Description: {doc.page_content}\n"
            f"LaTeX: {m['raw_latex']}\n"
            f"Suggested signature: {m['python_signature_hint']}\n"
            f"Source: {m['source_document']} p.{m['source_page']}\n"
            f"Variables: {m['variables_json']}\n"
        )
    return "\n\n".join(results)


def get_code_gen_agent_with_model(code_gen_model):
    def code_gen_agent(state: AgentState):
        messages = list(state["messages"])
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=CODE_GEN_SYSTEM_PROMPT)] + messages
        response = code_gen_model.invoke(messages)
        return {"messages": [response]}
    return code_gen_agent

def should_continue_code_gen(state: AgentState):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


def create_code_gen_agent(tools, gemma_model="gemma4:e4b"):
    code_gen_model = ChatOllama(model=gemma_model).bind_tools(tools)
    code_gen_agent = get_code_gen_agent_with_model(code_gen_model)
    code_gen_workflow = StateGraph(AgentState)
    code_gen_workflow.add_node("agent", code_gen_agent)
    code_gen_workflow.add_node("tools", ToolNode(tools))
    code_gen_workflow.add_edge(START, "agent")
    code_gen_workflow.add_conditional_edges("agent", should_continue_code_gen)
    code_gen_workflow.add_edge("tools", "agent")

    code_gen_app = code_gen_workflow.compile(checkpointer=MemorySaver())
    return code_gen_app


if __name__ == "__main__":
    code_gen_tools = [retrieve_implementable_formulas]
    code_gen_app = create_code_gen_agent(code_gen_tools)
    config = {"configurable": {"thread_id": "codegen_1"}}
    query = input("Hello I am an agent to generate code based on your questions and my provided papers/books. Please input a query:\n")
    result = code_gen_app.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config
    )
    logger.info(result["messages"][-1].content)