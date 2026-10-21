# ESP32 Physiological Sensor Node Circuit & Wiring Specification

## 1. Overview
The hardware component of the **AI-Based Multimodal Health Screening System** uses an **ESP32 DevKit v1** microcontroller interfaced with two non-invasive physiological medical sensors over a shared Inter-Integrated Circuit (I2C) two-wire bus:
1. **MAX30102**: Optical photoplethysmography (PPG) pulse oximetry and heart rate monitor.
2. **MLX90614**: Non-contact infrared temperature sensor (or DS18B20 digital skin contact sensor).

```
                      +-----------------------------+
                      |         ESP32 DevKit        |
                      |                             |
                      | 3V3 ------------------------+----> 3.3V VCC (All Sensors)
                      | GND ------------------------+----> Common Ground
                      | GPIO 21 (SDA) --------------+----> Shared I2C Data Line
                      | GPIO 22 (SCL) --------------+----> Shared I2C Clock Line
                      +-----------------------------+
                                     |
              +----------------------+----------------------+
              |                                             |
    +-------------------+                         +-------------------+
    |     MAX30102      |                         |     MLX90614      |
    |  Pulse Oximeter   |                         |   IR Temperature  |
    |  I2C Addr: 0x57   |                         |   I2C Addr: 0x5A  |
    +-------------------+                         +-------------------+
```

---

## 2. Pin Connections Table

| Sensor Component | Sensor Pin | ESP32 GPIO Pin | Function / Description |
| :--- | :--- | :--- | :--- |
| **MAX30102** | VIN / VCC | **3V3** | 3.3V DC Regulated Power |
| **MAX30102** | GND | **GND** | System Ground |
| **MAX30102** | SCL | **GPIO 22** | I2C Clock Line |
| **MAX30102** | SDA | **GPIO 21** | I2C Data Line |
| **MAX30102** | INT | *Not Connected* | Active-low hardware interrupt (optional) |
| **MLX90614** | VIN | **3V3** | 3.3V DC Regulated Power |
| **MLX90614** | GND | **GND** | System Ground |
| **MLX90614** | SCL | **GPIO 22** | Shared I2C Clock Line |
| **MLX90614** | SDA | **GPIO 21** | Shared I2C Data Line |

---

## 3. Pull-Up Resistor Requirements
- I2C lines (`SDA` and `SCL`) require $4.7\,\text{k}\Omega$ pull-up resistors to $3.3\text{V}$.
- Most commercial MAX30102 and MLX90614 breakout boards already integrate onboard $4.7\,\text{k}\Omega$ SMD pull-up resistors.
- Ensure operating voltage does **NOT** exceed $3.3\text{V}$ to protect ESP32 I/O buffers.

---

## 4. Software Dependencies (Arduino IDE)
1. In Arduino IDE `Library Manager`, install:
   - `ArduinoJson` (v6 or v7) by Benoit Blanchon
   - `SparkFun MAX3010x Pulse and Proximity Sensor Library`
   - `Adafruit MLX90614 Library`
2. Select Board: `ESP32 Dev Module`.
3. Set Upload Speed: `921600` baud.
4. Set Serial Monitor: `115200` baud.

---

## 5. Network & REST Ingestion Protocol
The ESP32 sends HTTP POST requests formatted as JSON to:
`POST http://<SERVER_IP>:5000/api/sensors/measurement`

```json
{
  "session_id": "c1f7a0b3-9821-4f4a-b5e2-632f1839db01",
  "heart_rate": 74.5,
  "spo2": 98.2,
  "temperature": 36.65,
  "is_simulated": false
}
```

The Flask server returns `HTTP 201 Created` with a `sensor_quality_score` (0-100%).
