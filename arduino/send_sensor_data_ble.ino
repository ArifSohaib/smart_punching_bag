#include <Arduino_LSM9DS1.h>
#include <ArduinoBLE.h>

BLEService sensorService("19B10000-E8F2-537E-4F6C-D104768A1214");

// One string characteristic for JSON
BLECharacteristic accelerationJSONChar(
  "19B10001-E8F2-537E-4F6C-D104768A1214",
  BLERead | BLENotify,
  100 // max length in bytes of the JSON string
);

float accX = 0.0;
float accY = 0.0;
float accZ = 0.0;

void setup() {
  Serial.begin(115200);
  #if defined(DEBUG)
  while (!Serial);
  #endif

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  if (!BLE.begin()) {
    Serial.println("Starting BLE failed!");
    while (1);
  }

  BLE.setLocalName("Nano33BLE_JSON");
  BLE.setAdvertisedService(sensorService);

  sensorService.addCharacteristic(accelerationJSONChar);
  BLE.addService(sensorService);

  BLE.advertise();
  Serial.println("BLE JSON IMU active...");
}

void loop() {
  BLEDevice central = BLE.central();

  if (central) {
    Serial.print("Connected to ");
    Serial.println(central.address());

    while (central.connected()) {
      if (IMU.accelerationAvailable()) {
        IMU.readAcceleration(accX, accY, accZ);

        // Build JSON string
        char jsonBuffer[100];
        snprintf(jsonBuffer, sizeof(jsonBuffer),
                 "{\"x\":%.4f,\"y\":%.4f,\"z\":%.4f}",
                 accX, accY, accZ);

        // Send as BLE notification
        accelerationJSONChar.writeValue(jsonBuffer);

        Serial.println(jsonBuffer); // debug
      }
      delay(50); // ~20Hz update
    }

    Serial.println("Disconnected");
  }
}
