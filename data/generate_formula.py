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
LAYER2_DB = Path(Path.cwd(),"chroma_db_layer2_gemma4:e4b")
embeddings = OllamaEmbeddings(model='embeddinggemma:300m')
layer2_store = Chroma(persist_directory=str(LAYER2_DB), embedding_function=embeddings)

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



class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

CODE_GEN_SYSTEM_PROMPT = """You are a code generation assistant specializing in 
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


def code_gen_agent(state: AgentState):
    messages = list(state["messages"])
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=CODE_GEN_SYSTEM_PROMPT)] + messages
    response = code_gen_model.invoke(messages)
    return {"messages": [response]}


def should_continue_code_gen(state: AgentState):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]



filtered_retriever = layer2_store.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": {
                "$and": [
                    {"type": {"$eq": "annotated_equation"}},
                    {"is_implementable": {"$eq": True}},
                    {"latex_quality": {"$ne": "low"}},  # skip garbled OCR
                ]
            }
        }
    )

simple_retriever = layer2_store.as_retriever()


query = "Write a function to calcaulate the average force of punches over a given time period"

simple_retrieved_docs = simple_retriever.invoke(query)
filtered_retrieved_docs = filtered_retriever.invoke(query)

print(f"simple retrieved docs = {len(simple_retrieved_docs)}")
print(f"filtered retrieved docs = {len(filtered_retrieved_docs)}")

for doc in simple_retrieved_docs:
    print(f"{doc.metadata}\n")

for doc in filtered_retrieved_docs:
    print(f"{doc.metadata}\n")

layer2_data = layer2_store.get(
    where={"type": "annotated_equation"},
    limit=20,
    include=["metadatas", "documents"]
)

for m in layer2_data["metadatas"]:
    print(f"impl={m['is_implementable']} | qual={m['latex_quality']} | latex={m['raw_latex'][:100]}")




code_gen_tools = [retrieve_implementable_formulas]
code_gen_model = ChatOllama(model="gemma4:e4b").bind_tools(code_gen_tools)



code_gen_workflow = StateGraph(AgentState)
code_gen_workflow.add_node("agent", code_gen_agent)
code_gen_workflow.add_node("tools", ToolNode(code_gen_tools))
code_gen_workflow.add_edge(START, "agent")
code_gen_workflow.add_conditional_edges("agent", should_continue_code_gen)
code_gen_workflow.add_edge("tools", "agent")

code_gen_app = code_gen_workflow.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "codegen_1"}}
result = code_gen_app.invoke(
    {"messages": [HumanMessage(content=query)]},
    config=config
)
print(result["messages"][-1].content)