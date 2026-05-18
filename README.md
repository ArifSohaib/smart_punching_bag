# Smart Punching Bag

Smart punching bag using gemma4 local llm with Ollama


### Environment Setup
install all the requirements using the [pyproject.toml](`pyproject.toml`)
Use `uv sync` in the folder to do the install
You may need to also install tessaract for ocr

Project has been tested on 32 GB RAM and AMD 7900 GRE 16 GB VRAM and AMD 9700x CPU.


### Putting the IMU code on Arduino Nano
Use the Arduino IDE (either application or web version) to upload the code to an Arduino Nano 33 BLE Sense

You don't need an actual punching bag, shaking the arduino will produce some results suitable for testing. 


### Run Data Gathering Application
Go to the app folder and then run
```bash
uv run fastapi run main.py
```

### Running data gathering from the docs 
- When the application has started go to localhost:8000/docs
- move the punching bag or remove the sensor and move it along with the battery to turn the sensor on
- make sure that bluetooth connection is turned on 
- then click /sensor/start and click Execute
- do the workout or test
- to stop the data gathering go to /sensor/stop and click Execute 

### SQL Tests
To test that the session has been recorded correctly and that gemma is able to query it, go to the `app` directory and run

```bash
uv run llm_tools.function_tests.py
```

## Ingesting the research papers
To ingest the data into Chroma db, from the root folder run

```bash
 uv run code_generation/data_prep.py
```

Once this is complete, you will see extracted images from the papers in the [data/extracted_images folder](data/extracted_images/) 

And you will see a new chroma_db folder in the data folder. This is layer 1

Then to populate layer 2, run:
```bash
uv run code_generation/formula_extraction.py
```

This will create another chroma db folder which will have information from all images from the previous stage.

Then to query this data and potentially generate code run

```bash
uv run code_generation/generate_formula.py

```

To use the full 26b model answer y or yes to the first prompt and then ask questions about punch measurements or human kinematics measured with IMUs.

Some example questions:
- How do I classify a punch from time-series IMU data using a learned model?
- Convert raw IMU accelerometer readings into impact force on the bag
- How do I extract features from a window of accelerometer samples for punch classification?