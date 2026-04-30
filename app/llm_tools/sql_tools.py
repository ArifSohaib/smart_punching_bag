from langchain_ollama import ChatOllama
from pathlib import Path 
import duckdb 
from sqlalchemy import create_engine, text 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_classic.chains import create_sql_query_chain

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

def generate_query_with_ollama(file_path:str, question:str):
    """
    Uses Ollama and QuerySQLDatabaseTool to generate a SQL query based on the user's natural language question.
    Args:
        file_path (str): path to the parquet file
        question (str): user's natural language question
    Returns:
        query: Generated SQL query as a string
    """
    _, engine = get_sql_connection(file_path)
    db = SQLDatabase(engine=engine)
    llm = ChatOllama(model="gemma4:e4b", temperature=0.1)
    sql_query_gen_chain = create_sql_query_chain(llm, db)
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
    full_sql_gen_chain = sql_query_gen_chain | clean_sql_chain |  StrOutputParser()
    response = full_sql_gen_chain.invoke(
        {"question": question})

    return response


if __name__ == "__main__":
    parquet_file = Path("data", "raw_punches.parquet")
    if not parquet_file.exists():
        print(f"Parquet file not found at {parquet_file}. Please run a workout session to generate data.")
    query = generate_query_with_ollama(str(parquet_file), "What is the average punch speed?")
    print(f"Generated Query:\n{query}")
    _ , engine = get_sql_connection(parquet_file)
    db = SQLDatabase(engine=engine)
    result = db.run(query)
    print("generated query results:")
    print(result)

        