CHARACTERISTIC_UUID = "19B10001-E8F2-537E-4F6C-D104768A1214"
DEVICE_NAME = "Nano33BLE_JSON"
from app.routers import sensor
from bleak import BleakClient
import json 
import asyncio 
from app.services.audio import play_vibe_instant
import logging 
import random
import numpy as np 
from datetime import datetime,timedelta 
from bleak import BleakScanner
logger = logging.getLogger("uvicorn.error")

random.seed(42)
vibe_outputs = ["No! Stop it! Mercy!"
"Please... it's too much.",
"No... please, stop!",
"Mercy! No! Please, stop!",
"Stop! It's fake! Mercy!",
"Mercy! Please, I'm done!",
"Stop! Please, have mercy.",
"Make it stop! Please."]

class BLEManager:
    def __init__(self):
        self.client = None
        self.is_running = False
        #set time delta 15 seconds ago so the first time the magnitude exceeds the threshold, it always plays the sound
        self.last_played = datetime.now() - timedelta(seconds=16)

    async def notification_handler(self, sender, data):
        """Processes incoming BLE strings and pushes to the active buffer."""
        logger.info(f"getting from sender  {sender}: {data}")
        if sensor.active_session:
            try:
                # Decode the bytes from Arduino
                decoded_data = json.loads(data.decode('utf-8'))
                sensor.active_session.add_point(
                    decoded_data['x'], 
                    decoded_data['y'], 
                    decoded_data['z']
                )
                mag = np.sqrt((decoded_data['x']**2) + (decoded_data['y']**2) + (decoded_data['z']**2))
                time_since_last = datetime.now() - self.last_played
                if mag> 5 and time_since_last > timedelta(seconds=5):
                    quip = random.choice(vibe_outputs)
                    play_vibe_instant(quip)
                    self.last_played = datetime.now()
            except Exception as e:
                logger.error(f"Parsing error: {e}")

    async def connect_and_run(self):
        """Scans, connects, and listens for the Arduino."""
        logger.info("Scanning for Smart Bag...")
        device = await BleakScanner.find_device_by_name(DEVICE_NAME)
        
        if not device:
            logger.error("Bag not found. Make sure it's powered (punch it to power it on!).")
            return
        logger.info(f"Found {DEVICE_NAME}, {device.address} - attempting to connect...")

        async with BleakClient(device) as client:
            self.client = client
            logger.info(f"Connected to {device.name}")
            
            # Start notifications
            await client.start_notify(CHARACTERISTIC_UUID, self.notification_handler)
            
            while self.is_running and client.is_connected:
                await asyncio.sleep(0.5) 


            #when NOT running
            logger.info("Stopping BLE session...")
            if client.is_connected:
                try:
                    await client.stop_notify(CHARACTERISTIC_UUID)
                except Exception as e:
                    logger.error(f"Error stopping notification {e}")
            self.client = None 
            logger.info(f"BLE disconnected cleanly")