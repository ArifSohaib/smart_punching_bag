#generate audio from sensor readings
#do it in the background so the data collection is not impacted

import os 
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import subprocess
from collections import deque
import logging
logger = logging.getLogger("uvicorn.error")
                           

llm = ChatOllama(model="gemma4:e4b")

vibe_pool = deque(["No! Stop it! Mercy!"
"Please... it's too much."
"No... please, stop!",
"Mercy! No! Please, stop!",
"Stop! It's fake! Mercy!",
"Mercy! Please, I'm done!",
"Stop! Please, have mercy.",
"Make it stop! Please."])
vibe_pool_bad = [
 'Weak. Hit me harder!',
 "Harder! You can't stop!",
 'Weak. Hit me again.',
 "More! You can't...",
 'Too weak. Hit me harder.',
 'More! You can do better!',
 'Harder! You can do better.',
 'Harder! Make it huge!']

def refresh_vibe_pool():
    """Run this in a slow background loop to keep the quips fresh."""
    global vibe_pool
    # Only generate if the pool is getting low
    if len(vibe_pool) < 10:
        new_quip = generate_reaction(5, 5) # Your existing LLM function
        vibe_pool.append(new_quip)


punch_vibe_prompt = f"""
You are being punched in the face. 
Based on the force (magnitude) and the snap (delta), give a very short (under 5 words) reaction.
- High Delta (> 4.5): You're hurt by the punch, beg for mercy
- Low Delta (<2.0): You try to mock the puncher and ask for more. 
""" 

def generate_reaction(magnitude, delta):
    prompt = f"""Impact Magnitude: {magnitude}, Snap Delta: {delta}. Reaction?"""
    response = llm.invoke([("system", punch_vibe_prompt), ("user", prompt)])
    return response.content.strip().replace('"', '')

def play_text(text):
    logger.info(f"BAG SAYS: {text}")
    try:
        # -t male1: Sets a male voice
        # -p 20: Pitch (slightly higher for a 'sentient' feel)
        # -r 10: Speed (slightly faster)
        # & ensures it doesn't block your FastAPI execution
        os.system(f'spd-say -t male1 -p 20 -r 10 "{text}" &')
    except Exception as e:
        logger.info(f"Audio Error: {e}")

def play_vibe_instant(text):
    logger.info(f"!!! BAG SAYS: {text} !!!")
    os.system(f'spd-say -t male1 -p 20 -r 15 "{text}" &')