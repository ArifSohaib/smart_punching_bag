#holds the PyArrow Schema for saving to Parquet + DuckDB

import pyarrow as pa
import pyarrow.parquet as pq
import uuid
from datetime import datetime
import numpy as np

# Raw Sensor Schema (Every punch/shake)
raw_schema = pa.schema([
    ("timestamp", pa.timestamp('ms')),
    ("session_id", pa.string()),
    ("x", pa.float32()),
    ("y", pa.float32()),
    ("z", pa.float32()),
    ("magnitude", pa.float32())
])

# Round Schema (One row per time the bag wakes up)
round_schema = pa.schema([
    ("session_id", pa.string()),
    ("start_time", pa.timestamp('ms')),
    ("punch_count", pa.int32()),
    ("peak_force", pa.float32())
])