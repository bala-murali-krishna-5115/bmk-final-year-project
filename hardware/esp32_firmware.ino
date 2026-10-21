/*
 * ======================================================================================
 * AI-Based Multimodal Health Screening System - ESP32 Firmware
 * 
 * Target Microcontroller: ESP32 DevKit v1 / NodeMCU-32S
 * Sensors:
 *   1. MAX30102 Heart Rate & SpO2 Sensor (I2C: SDA=GPIO 21, SCL=GPIO 22)
 *   2. MLX90614 Contactless Infrared Temperature Sensor (I2C: Shared bus 0x5A)
 * 
 * Payload Destination: POST http://<SERVER_IP>:5000/api/sensors/measurement
 * 
 * DISCLAIMER: Academic research prototype. NOT for clinical medical diagnosis.
 * ======================================================================================
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <ArduinoJson.h>

// WiFi Configuration
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// Backend Server Configuration
const char* SERVER_ENDPOINT = "http://192.168.1.100:5000/api/sensors/measurement";
String CURRENT_SESSION_ID   = "S_ESP32_DEFAULT"; // Updated dynamically or via Serial

// Timing Configuration
const unsigned long TRANSMIT_INTERVAL_MS = 1000; // 1 Hz sample transmission
unsigned long lastTransmitTime = 0;

// Hardware Status LED Pin
const int STATUS_LED = 2; // Built-in Blue LED

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n[ESP32] Initializing Multimodal Hardware Sensor Node...");

  pinMode(STATUS_LED, OUTPUT);
  digitalWrite(STATUS_LED, LOW);

  // Initialize I2C Bus (SDA=21, SCL=22)
  Wire.begin(21, 22);
  Serial.println("[ESP32] I2C Bus initialized on GPIO 21 (SDA), GPIO 22 (SCL).");

  // Connect to WiFi
  connectWiFi();

  digitalWrite(STATUS_LED, HIGH);
  Serial.println("[ESP32] Sensor node ready for transmission.");
}

void loop() {
  // Check WiFi connection
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }

  // Handle Serial Commands (e.g., SET_SESSION:<session_id>)
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.startsWith("SET_SESSION:")) {
      CURRENT_SESSION_ID = cmd.substring(12);
      Serial.println("[ESP32] Active Session ID set to: " + CURRENT_SESSION_ID);
    }
  }

  // Periodic Telemetry Transmission
  if (millis() - lastTransmitTime >= TRANSMIT_INTERVAL_MS) {
    lastTransmitTime = millis();
    transmitSensorData();
  }
}

void connectWiFi() {
  Serial.print("[WiFi] Connecting to: ");
  Serial.println(WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    digitalWrite(STATUS_LED, !digitalRead(STATUS_LED));
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[WiFi] Connected successfully!");
    Serial.print("[WiFi] IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\n[WiFi] Connection timeout. Retrying in loop...");
  }
}

void transmitSensorData() {
  // Read physical physiological sensors or fallback to realistic test bounds
  float heartRate = readHeartRate();
  float spo2Val   = readSpO2();
  float bodyTemp  = readBodyTemperature();
  bool isSimulated = false;

  // If sensor readouts indicate unattached sensor, flag simulated
  if (heartRate <= 0.0 || spo2Val <= 0.0) {
    heartRate = random(700, 780) / 10.0;     // 70.0 - 78.0 bpm
    spo2Val   = random(975, 995) / 10.0;     // 97.5 - 99.5 %
    bodyTemp  = random(3640, 3680) / 100.0;  // 36.40 - 36.80 °C
    isSimulated = true;
  }

  // Construct JSON Payload
  StaticJsonDocument<256> doc;
  doc["session_id"]   = CURRENT_SESSION_ID;
  doc["heart_rate"]   = heartRate;
  doc["spo2"]         = spo2Val;
  doc["temperature"]  = bodyTemp;
  doc["is_simulated"] = isSimulated;

  String jsonPayload;
  serializeJson(doc, jsonPayload);

  // Send HTTP POST Request
  HTTPClient http;
  http.begin(SERVER_ENDPOINT);
  http.addHeader("Content-Type", "application/json");

  int httpCode = http.POST(jsonPayload);

  if (httpCode > 0) {
    Serial.printf("[HTTP] POST Result: %d | Sent: %s\n", httpCode, jsonPayload.c_str());
  } else {
    Serial.printf("[HTTP] POST Failed. Error: %s\n", http.errorToString(httpCode).c_str());
  }

  http.end();
}

// Sensor Driver Stubs (Integrates with SparkFun_MAX3010x and Adafruit_MLX90614 libraries)
float readHeartRate() {
  // Return pulse in bpm from MAX30102 buffer
  return -1.0; // Trigger fallback or bind to MAX30102 sensor.getHeartRate()
}

float readSpO2() {
  // Return SpO2 percentage from MAX30102 algorithm
  return -1.0; // Trigger fallback or bind to MAX30102 sensor.getSpO2()
}

float readBodyTemperature() {
  // Return temperature in Celsius from MLX90614 sensor.readObjectTempC()
  return -1.0; // Trigger fallback or bind to MLX90614
}
