# Smart Punching Bag

Smart punching bag using local llm. 

details to follow later

from app folder

### Run Data Gathering Application
```bash
fastapi run main.py
```


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