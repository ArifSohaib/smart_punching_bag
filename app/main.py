from fastapi import FastAPI, BackgroundTasks
import uuid
from app.routers import sensor
from app.core.buffer import PunchBuffer
from app.core.ble_manager import BLEManager
import logging 
logger = logging.getLogger("uvicorn.error")

ble_manager = BLEManager()


app = FastAPI()
app.router.include_router(sensor.router)

@app.post("/workout/start")
async def start_workout(background_tasks: BackgroundTasks):
    if ble_manager.is_running:
        return {"status": "Already running"}
    
    # 1. Setup the storage session
    sensor.active_session = PunchBuffer(session_id=str(uuid.uuid4()))
    
    # 2. Start the BLE loop in the background
    ble_manager.is_running = True
    background_tasks.add_task(ble_manager.connect_and_run)
    
    return {"message": "Scanning for bag and starting workout..."}

@app.post("/workout/stop")
async def stop_workout():
    ble_manager.is_running = False
    logger.info(f"set ble_manager.is_running = False")
    if sensor.active_session:
        sensor.active_session.flush_to_parquet()
        sensor.active_session = None
    return {"message": "Workout stopped and data saved."}