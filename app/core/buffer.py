from datetime import datetime 
from .schemas import raw_schema, round_schema
import pyarrow as pa 
import pyarrow.parquet as pq 
from pathlib import Path
import numpy as np 
import logging 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("uvicorn.error")

class PunchBuffer:
    def __init__(self, session_id):
        self.session_id = session_id
        self.data = []
        self.start_time = datetime.now()

    def add_point(self, x, y, z):
        mag = np.sqrt(x**2 + y**2 + z**2)
        self.data.append({
            "timestamp": datetime.now(),
            "session_id": self.session_id,
            "x": x, "y": y, "z": z,
            "magnitude": mag
        })

    def flush_to_parquet(self, filename="raw_punches.parquet"):
        if not self.data: return
        table = pa.Table.from_pylist(self.data, schema=raw_schema)
        if Path(filename).exists():
            existing_table = pq.read_table(filename)
            combined = pa.concat_tables([existing_table, table])
            pq.write_table(combined, filename)
        else:
            pq.write_table(table, filename) 
        logger.info(f"Saved {len(self.data)} points to {filename}")
