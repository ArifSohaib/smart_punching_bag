from langchain_ollama import ChatOllama
from pathlib import Path
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import logging

logger = logging.getLogger(__name__)


def get_sql_connection(file_path: str):
    """
    Returns the SQLAlchemy connection for the parquet file with duckdb as the dialect.

    Args:
        file_path (str): path to the parquet file

    Returns:
        tuple: (connection, engine) - the SQLAlchemy connection and engine
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")

    connection_string = f"duckdb:///{file_path}"
    engine = create_engine(connection_string)
    connection = engine.connect()

    # Create the raw_punches view if it doesn't exist
    result = connection.execute(
        text("SELECT name FROM sqlite_master WHERE type='view' AND name='raw_punches'")
    )
    if not result.fetchone():
        connection.execute(text(f"CREATE VIEW raw_punches AS SELECT * FROM '{file_path}'"))

    return connection, engine


def get_schema_string(engine: Engine) -> str:
    """
    Generate a schema description string by querying DuckDB directly.
    SQLAlchemy's inspect() fails on duckdb-engine because of pg_catalog quirks,
    so we use DuckDB's native DESCRIBE and information_schema queries instead.

    Args:
        engine: SQLAlchemy engine

    Returns:
        A multi-line string describing tables and their columns.
    """
    schema_parts = []
    with engine.connect() as conn:
        # Get all table and view names
        tables_result = conn.execute(text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ))
        table_names = [row[0] for row in tables_result.fetchall()]

        for table in table_names:
            # DuckDB-native DESCRIBE returns column info reliably
            cols_result = conn.execute(text(f"DESCRIBE {table}"))
            cols = cols_result.fetchall()
            col_descs = ", ".join(f"{row[0]} {row[1]}" for row in cols)
            schema_parts.append(f"Table {table}: {col_descs}")

    return "\n".join(schema_parts)


def execute_sql_query(engine: Engine, sql: str) -> str:
    """
    Executes a SQL query using the given engine and returns the results as a string.
    Replaces langchain_community.tools.QuerySQLDatabaseTool.

    Args:
        engine: SQLAlchemy engine
        sql: SQL query string

    Returns:
        Stringified list of result dicts, or an error message.
    """
    # Strip any markdown code fences the LLM might have added
    sql = sql.strip()
    if sql.startswith("```"):
        sql = sql.strip("`")
        if sql.lower().startswith("sql"):
            sql = sql[3:].strip()

    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            columns = result.keys()
            rows = [dict(zip(columns, row)) for row in result.fetchall()]
            return str(rows)
    except Exception as e:
        logger.error(f"SQL execution failed: {e}\nQuery was: {sql}")
        return f"SQL Error: {e}"


def query_parquet_with_sql(file_path: str, query: str) -> list[dict]:
    """
    Executes a SQL query against the parquet file using duckdb and returns the results.

    Args:
        file_path: path to the parquet file
        query: SQL query to execute

    Returns:
        Query results as a list of dictionaries
    """
    connection, _ = get_sql_connection(file_path)
    try:
        result = connection.execute(text(query))
        columns = result.keys()
        results = [dict(zip(columns, row)) for row in result.fetchall()]
        return results
    finally:
        connection.close()


def run_query_with_ollama(file_path: str, question: str) -> str:
    """
    Uses Ollama to generate a SQL query from natural language, then executes it.

    Args:
        file_path: path to the parquet file
        question: user's natural language question

    Returns:
        The response from the executed query as a string.
    """
    _, engine = get_sql_connection(file_path)
    schema = get_schema_string(engine)

    llm = ChatOllama(model="gemma4:e4b", temperature=0.1)

    sql_gen_prompt = ChatPromptTemplate.from_template("""
Given the following database schema, write a SQLite query to answer the user's question.
Schema: {schema}
Question: {question}
SQL Query:
""")

    clean_sql_prompt = ChatPromptTemplate.from_template("""
You are an expert in SQLite.
You are asked to fix badly formed SQLite queries,
which might contain unneeded prefixes or suffixes and only use known column names.
Given the following unclean SQL statement,
transform it to a clean, executable SQL statement for SQLite.
Always prefix column names with the table name.
Only return an executable SQL statement which terminates with a semicolon.
Do not return anything else.
Do not include the language name or symbols like ```.

There is one table called raw_punches with these columns:
session_id, x, y, z, and timestamp.
x, y, and z are floats.

Unclean SQL: {unclean_sql}
""")

    chain = (
        RunnablePassthrough.assign(schema=lambda _: schema)
        | sql_gen_prompt
        | llm
        | StrOutputParser()
        | (lambda x: {"unclean_sql": x})
        | clean_sql_prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(lambda sql: execute_sql_query(engine, sql))
    )

    response = chain.invoke({"question": question})
    return str(response)