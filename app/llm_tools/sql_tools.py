from langchain_ollama import ChatOllama
from pathlib import Path 
from sqlalchemy import create_engine, text 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool 

def get_sql_connection(file_path:str):
    """
    returns the SqlAlchemy connection for the parquet file with duckdb as the dialect 
    Args:
        file_path (str): path to the parquet file 
    returns:
        connection: SqlAlchemy connection 
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    connection_string = f"duckdb:///{file_path}"
    engine = create_engine(connection_string)
    connection = engine.connect()
    result = connection.execute(text("SELECT name FROM sqlite_master WHERE type='view' AND name='raw_punches'"))
    if not result.fetchone():
        connection.execute(text(f"CREATE VIEW raw_punches AS SELECT * FROM '{file_path}'"))
    return connection, engine 


def query_parquet_with_sql(file_path:str, query:str):
    """
    Executes a SQL query against the parquet file using duckdb and returns the results.
    Args:
        file_path (str): path to the parquet file
        query (str): SQL query to execute
    Returns:
        results: Query results as a list of dictionaries
    """
    connection, _ = get_sql_connection(file_path)
    result = connection.execute(text(query))
    columns = result.keys()
    results = [dict(zip(columns, row)) for row in result.fetchall()]
    connection.close()
    return results


def run_query_with_ollama(file_path:str, question:str)->str:
    """
    Uses Ollama and QuerySQLDatabaseTool to generate a SQL query based on the user's natural language question.
    Args:
        file_path (str): path to the parquet file
        question (str): user's natural language question
    Returns:
        response (str): the response from the executed query
    """
    _, engine = get_sql_connection(file_path)
    db = SQLDatabase(engine=engine)
    llm = ChatOllama(model="gemma4:e4b", temperature=0.1)
    sql_gen_prompt = ChatPromptTemplate.from_template("""
        Given the following database schema, write a SQLite query to answer the user's question.
        Schema: {schema}
        Question: {question}
        SQL Query:
    """)
    clean_sql_prompt_template = """
        You are an expert in SQLite. 
        You are asked to fix badly formed SQLite queries, 
        which might contain unneeded prefixes or suffixes and only use known column names. 
        Given the following unclean SQL statement, 
        transform it to a clean, 
        executable SQL statement for SQLite.
        Always prefix column names with the table name.
        Only return an executable SQL statement which terminates 
        with a semicolon. Do not return anything else.
        Do not include the language name or symbols like ```.
        There is one table called raw_punches and it has columns:
        session_id, x, y, z, and timestamp.
        x, y and z are floats
        Unclean SQL: {unclean_sql}
    """
    clean_sql_prompt = ChatPromptTemplate.from_template(clean_sql_prompt_template)
    clean_sql_chain = clean_sql_prompt | llm
    sql_query_exec_tool = QuerySQLDatabaseTool(db=db)
    reject_alter_and_write_sql_prompt_template = """
        You are an expert in SQLite safety. If you see an SQL statement that deletes, modifies or writes to the database, you will reject it
        stating that the LLM should not be allowed to execute such statements. BUT you will return the SQL statement to allow the user to execute it themselves if they choose to.
        original SQL statement: {sql_statement}"""
    reject_alter_and_write_sql_prompt = ChatPromptTemplate.from_template(reject_alter_and_write_sql_prompt_template)
    reject_sql_chain = reject_alter_and_write_sql_prompt | llm #TODO: requires branching path using langchain
    full_sql_gen_chain = (
        RunnablePassthrough.assign(schema=lambda _: db.get_table_info())

        | sql_gen_prompt 
        | llm 
        | StrOutputParser()

        | (lambda x: {"unclean_sql": x}) 
        | clean_sql_prompt 
        | llm 

        | StrOutputParser()
        | sql_query_exec_tool 
    )

    response = full_sql_gen_chain.invoke({"question": question})

    return str(response)


        