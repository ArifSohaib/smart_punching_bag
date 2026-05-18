from fastapi import APIRouter, HTTPException
from app.models.punch_model import PunchUpdate
from app.core.buffer import PunchBuffer
import uuid
from app.services.audio import generate_reaction, play_text, play_vibe_instant, refresh_vibe_pool
from fastapi import BackgroundTasks
import numpy as np 
from app.core.ble_manager import BLEManager
import logging 
from pathlib import Path
logger = logging.getLogger("uvicorn.error")
ble_manager = BLEManager()

router = APIRouter(prefix="/sensor", tags=["Workout"])

active_session = None
last_magnitude = 0

@router.post("/start")
async def start_workout(background_tasks: BackgroundTasks):
    if ble_manager.is_running:
        return {"status": "Sensor already connected and active"}

    global active_session
    global last_magnitude

    session_id = str(uuid.uuid4())
    active_session = PunchBuffer(session_id=session_id)
    ble_manager.is_running = True
    background_tasks.add_task(ble_manager.connect_and_run)
    logger.info(f"Started workout session {session_id}")

    return {"message": "Scanning for bag and starting workout ...", "session_id": session_id}

@router.post("/update")
async def receive_punch(data: PunchUpdate, background_tasks: BackgroundTasks):
    if not active_session:
        # Silently ignore or return error if workout hasn't started
        return {"status": "no_active_session"}
    
    active_session.add_point(data.x, data.y, data.z, data.session_id)
    current_mag = np.sqrt((data.x ** 2) + (data.y ** 2 )+ (data.z ** 2))
    delta = np.round(current_mag - last_magnitude, 2)
    last_magnitude = current_mag
    if delta > 5:
        print("playing sound")
        play_vibe_instant()
    background_tasks.add_task(refresh_vibe_pool)
    return {"status": "recorded"}

@router.post("/stop")
async def stop_workout(background_tasks: BackgroundTasks):
    global active_session
    if not active_session:
        return {"error": "No session to stop"}
    ble_manager.is_running = False
    logger.info(f"Stopping workout session {active_session.session_id}")

    if active_session:
        filename_session = Path("data", f"workout_{active_session.session_id}.parquet")
        filename_raw = Path("data", "raw_punches.parquet")
        if not filename_raw.parent.exists():
            filename_raw.parent.mkdir(parents=True)
        background_tasks.add_task(active_session.flush_to_parquet, filename=filename_raw)
        logger.info(f"Workout data saved to {filename_raw}")
        active_session = None
        return {"message": "Workout stopped and saved", "file": filename_raw.name}
    else:
        return {"error": "No active session to save"}

async def process_vibe(mag, delta):
    reaction = generate_reaction(mag, delta)
    play_text(reaction)