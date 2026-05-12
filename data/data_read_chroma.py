from langchain.tools import tool
from langchain_ollama import ChatOllama

from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END 
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver 
import logging 
from pathlib import Path 
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from rich.markdown import Markdown
from rich.console import Console
from langchain.messages import HumanMessage, AIMessage, SystemMessage
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(name)s - %(filename)s: %(message)s")

logger = logging.getLogger(__name__)
chroma_path = Path.cwd()
chroma_file = "chroma_db"
chroma_db_path = Path(chroma_path, chroma_file)
gemma_model = "gemma4:e4b" 
llm = ChatOllama(model=gemma_model)

embeddings = OllamaEmbeddings(model='embeddinggemma:300m')
vector_store = Chroma(
    persist_directory=str(chroma_db_path),
    embedding_function=embeddings
)
@tool
def retrieve_research_papers(query: str) -> str:
    """ALWAYS call this first. Search the research paper database and return 
    relevant passages to answer the user's question.
    
    Args:
        query: The user's question or key concepts to search for, 
               passed exactly as the user asked it.
    Returns:
        Relevant passages from the research papers with metadata.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})
    docs = retriever.invoke(query)
    seen = set()
    unique_docs = []
    for doc in docs:
        content_hash = hash(doc.page_content)
        if content_hash not in seen:
            seen.add(content_hash)
            unique_docs.append(doc)
    results = []
    for doc in unique_docs:
        results.append(
            f"[Source: {doc.metadata.get('document')} | "
            f"Page: {doc.metadata.get('page_no')} | "
            f"Section: {doc.metadata.get('section_h2', 'N/A')}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(results)
tools = [retrieve_research_papers] 
model = ChatOllama(model=gemma_model).bind_tools(tools)


class AgentState(TypedDict):
    # This automatically handles message history append logic
    messages: Annotated[Sequence[BaseMessage], add_messages]

SYSTEM_PROMPT = """You are a specialized research assistant with access to biomechanics and IMU research papers.

IMPORTANT RULES:
1. You MUST call the retrieve_research_papers tool before answering ANY question.
2. Base your answer ONLY on the retrieved content. Do not use your general knowledge.
3. If the retrieved content does not contain enough information to answer, say so explicitly.
4. Always reference the source document and page number from the metadata when possible.
5. When formulas are present in the retrieved content, include them in your answer exactly as written.
"""



tools = [retrieve_research_papers] 
model = ChatOllama(model=gemma_model).bind_tools(tools)


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    # If last message is AI with no tool calls, we're done
    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
        return END
    # If last message is AI with tool calls, execute them
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END

def must_retrieve(state: AgentState) -> AgentState:
    """Directly invoke the retrieval tool without going through the model."""
    # Get the last human message to use as the query
    last_human = next(
        m for m in reversed(state["messages"]) 
        if isinstance(m, HumanMessage)
    )
    
    # Call the tool directly — no model involved
    retrieved_content = retrieve_research_papers.invoke(last_human.content)
    
    # Inject retrieved content as a system message so the model sees it
    retrieval_message = SystemMessage(
        content=f"Retrieved context from research papers:\n\n{retrieved_content}"
    )
    return {"messages": [retrieval_message]}

def call_model(state: AgentState) -> AgentState:
    messages = list(state["messages"])
    if not any(
        isinstance(m, SystemMessage) and "You are a specialized" in m.content 
        for m in messages
    ):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = model.invoke(messages)
    return {"messages": [response]}

workflow = StateGraph(AgentState)
workflow.add_node("retrieve", must_retrieve)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))  # still available for follow-up tool calls

workflow.add_edge(START, "retrieve")         # always retrieve first
workflow.add_edge("retrieve", "agent")       # pass context straight to model
workflow.add_conditional_edges("agent", should_continue)  # loop if model calls more tools
workflow.add_edge("tools", "agent")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "test_1"}}
console = Console()
def chat_debug(user_input: str):
    result = app.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
    
    # Print all messages to see tool calls and retrieved content
    for msg in result["messages"]:
        msg_type = type(msg).__name__
        console.print(Markdown(f"\n[bold yellow]{msg_type}[/bold yellow]"))
        
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            console.print(Markdown(f"[cyan]Tool called: {msg.tool_calls}[/cyan]"))
        
        console.print(Markdown(msg.content))

chat_debug("What is a quaternion and how is it used in kinematic calculations?")
