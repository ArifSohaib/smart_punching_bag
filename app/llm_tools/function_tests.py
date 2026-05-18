import sys 
sys.path.append(".")
import llm_tools.sql_tools
from sqlalchemy import text
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(filename)s: %(message)s"
)
logger = logging.getLogger(__name__)


def test_sql_connection(file_path: str):
    """
    Tests the SQL connection by running a simple query to read the first 5 rows.

    Args:
        file_path: path to the parquet file
    """
    logger.info(f"Testing SQL connection on {file_path}")
    try:
        connection, _ = llm_tools.sql_tools.get_sql_connection(file_path)
        with connection as conn:
            result = conn.execute(text("SELECT * FROM raw_punches LIMIT 5"))
            for row in result:
                logger.info(row)
    except Exception as e:
        logger.error(f"Error testing SQL connection: {e}")


def test_schema_string(file_path: str):
    """
    Tests the schema string generator — useful for verifying what the LLM sees.

    Args:
        file_path: path to the parquet file
    """
    logger.info("Testing schema string generation")
    try:
        _, engine = llm_tools.sql_tools.get_sql_connection(file_path)
        schema = llm_tools.sql_tools.get_schema_string(engine)
        logger.info(f"Generated schema:\n{schema}")
    except Exception as e:
        logger.error(f"Error generating schema string: {e}")


def test_direct_query(file_path: str, query: str):
    """
    Tests direct SQL execution without LLM involvement.

    Args:
        file_path: path to the parquet file
        query: a raw SQL query
    """
    logger.info(f"Testing direct query: {query}")
    try:
        results = llm_tools.sql_tools.query_parquet_with_sql(file_path, query)
        logger.info(f"Got {len(results)} rows")
        for row in results[:5]:
            logger.info(row)
    except Exception as e:
        logger.error(f"Error executing direct query: {e}")


def test_query_generation(file_path: str, question: str):
    """
    Tests the SQL query generation using Ollama with a natural language question.

    Args:
        file_path: path to the parquet file
        question: natural language question to generate SQL for
    """
    logger.info(f"Testing LLM query generation: {question}")
    try:
        result = llm_tools.sql_tools.run_query_with_ollama(str(file_path), question)
        logger.info(f"Result: {result}")
    except Exception as e:
        logger.error(f"Error in LLM query generation: {e}")


if __name__ == "__main__":
    parquet_file = Path("data", "raw_punches.parquet")
    if not parquet_file.exists():
        raise FileNotFoundError(
            f"Parquet file not found at {parquet_file}. "
            "Please run a workout session to generate data."
        )

    # Stage 1: verify the connection works
    test_sql_connection(str(parquet_file))

    # Stage 2: verify schema introspection works
    test_schema_string(str(parquet_file))

    # Stage 3: verify direct SQL execution
    test_direct_query(
        str(parquet_file),
        "SELECT session_id, COUNT(*) as punch_count FROM raw_punches GROUP BY session_id"
    )

    # Stage 4: LLM-generated queries
    questions = [
        "What are the average punch magnitudes for each session?",
        "How many punches were thrown in each session?",
        "Calculate the change in magnitude (gradient) of each punch and store it in a column called delta_magnitude",
    ]

    for question in questions:
        test_query_generation(str(parquet_file), question)