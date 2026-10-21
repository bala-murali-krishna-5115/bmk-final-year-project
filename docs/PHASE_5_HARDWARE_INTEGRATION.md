# Phase 5: Hardware Sensor Node & ESP32 Integration

## 1. Overview & System Scope
Phase 5 integrates physical physiological sensors with the central backend server using an **ESP32 DevKit** micro-controller over WiFi.

> **ACADEMIC DISCLAIMER**:
> Sensors provide preliminary vitals sampling for pattern screening. It does not constitute medical monitoring.

---

## 2. Hardware Architecture & Sensors
- **Microcontroller**: ESP32 DevKit v1 (Tensilica Xtensa Dual-Core 240 MHz, 520 KB SRAM).
- **Sensors**:
  1. **MAX30102**: Optical photoplethysmography (PPG) module with integrated red and IR LEDs for Heart Rate (bpm) and SpO2 (%).
  2. **MLX90614**: Contactless infrared thermometer measuring forehead/skin surface temperature (°C).

---

## 3. Communication Protocol & Telemetry Schema
- **Physical Bus**: Shared I2C (SDA = GPIO 21, SCL = GPIO 22, 3.3V DC power).
- **Network Protocol**: HTTP 1.1 REST JSON POST over local 802.11 b/g/n WiFi.
- **Payload Schema**:
  ```json
  {
    "session_id": "c1f7a0b3-9821-4f4a-b5e2-632f1839db01",
    "heart_rate": 74.0,
    "spo2": 98.2,
    "temperature": 36.65,
    "is_simulated": false
  }
  ```
- **Ingestion Endpoint**: `POST /api/sensors/measurement`
- **Response**: `HTTP 201 Created` with `sensor_quality_score` (0-100%) and validation warnings.

---

## 4. Sensor Quality Scoring Engine
The backend evaluates incoming physiological values against known human physiological bounds:
- **Heart Rate**: Valid range [40, 200] bpm. Values outside penalize score by -40%.
- **SpO2**: Valid range [70, 100] %. Values outside penalize score by -40%.
- **Temperature**: Valid range [30.0, 43.0] °C. Values outside penalize score by -35%.

If the resulting sensor quality score combined with camera quality falls below **65%**, screening analysis is strictly refused.

---

## 5. Software Simulation & Demonstration Mode
To facilitate evaluations and presentations where physical ESP32 hardware cannot be plugged in:
- Built-in simulator endpoint: `POST /api/sensors/simulate` with modes (`normal`, `elevated`, `fatigue`, `low_quality`).
- Interactive simulator CLI: `python -m hardware.simulator --session-id <ID> --mode normal`.
- Web dashboard live dropdown toggle.
