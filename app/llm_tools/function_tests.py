import llm_tools.sql_tools
from sqlalchemy import text
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDatabaseTool
from pathlib import Path 


def test_sql_connection(file_path:str):
    """
    Tests the SQL connection by running a simple query to read the first 5 rows of the parquet file.
    Args:
        file_path (str): path to the parquet file 
    """
    try:
        connection, _ = llm_tools.sql_tools.get_sql_connection(file_path)
        
        with connection as conn:
            result = conn.execute(text("SELECT * FROM raw_punches LIMIT 5"))
            for row in result:
                print(row)
    except Exception as e:
        print(f"Error testing SQL connection: {e}")


def test_query_generation(file_path:str, question:str):
    """
    Tests the SQL query generation using Ollama by providing a natural language question.
    Args:
        file_path (str): path to the parquet file 
        question (str): natural language question to generate SQL for
    """
    result = llm_tools.sql_tools.run_query_with_ollama(str(file_path), question)
    print(result)


if __name__ == "__main__":
    parquet_file = Path("data", "raw_punches.parquet")
    if not parquet_file.exists():
        raise Exception(f"Parquet file not found at {parquet_file}. Please run a workout session to generate data.")
    test_sql_connection(str(parquet_file))

    questions = []
    questions.append("What are the average punch magnitudes for each session?")
    questions.append("How many punches were thrown in each session?")
    questions.append("Calculate the change in magnitude (gradient) of each punch and store it in a column called delta_magnitude")

    for question in questions:
        test_query_generation(str(parquet_file), question)
