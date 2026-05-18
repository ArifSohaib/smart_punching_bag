# Smart Punching Bag

Smart punching bag using gemma4 local llm with Ollama


### Environment Setup
install all the requirements using the [pyproject.toml](`pyproject.toml`)
Use `uv sync` in the folder to do the install
You may need to also install tessaract for ocr

Project has been tested on 32 GB RAM and AMD 7900 GRE 16 GB VRAM and AMD 9700x CPU.

### Papers
I can't put the papers directly in the repo since they are sourced from

### Putting the IMU code on Arduino Nano
Use the Arduino IDE (either application or web version) to upload the code to an Arduino Nano 33 BLE Sense

You don't need an actual punching bag, shaking the arduino will produce some results suitable for testing. 


### Run Data Gathering Application
Go to the app folder and then run
```bash
uv run fastapi run main.py
```
This will start a fastapi server to gather data. 

Ensure that Bluetooth is turned on on your desktop and then in the docs click start
This starts the recording of the data. Continue recording until a workout session is done. 

when you are done scroll down in the fastapi swagger docs and click stop

If you do another recording the previous one will not be overwritten and instead it will be stored as a new session. 





### Running data gathering from the docs 
- When the application has started go to localhost:8000/docs
- move the punching bag or remove the sensor and move it along with the battery to turn the sensor on
- make sure that bluetooth connection is turned on 
- then click /sensor/start and click Execute
- do the workout or test
- to stop the data gathering go to /sensor/stop and click Execute 

### Tool Tests
```bash
python -m llm_tools.function_tests
```