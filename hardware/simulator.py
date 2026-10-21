import time
import random
import requests
import argparse
import sys

def run_simulator(session_id: str, server_url: str = "http://localhost:5000", mode: str = "normal", interval: float = 1.0, count: int = 10):
    """
    Simulates ESP32 physiological sensor telemetry streaming to backend REST API.
    """
    endpoint = f"{server_url.rstrip('/')}/api/sensors/measurement"
    print(f"\n[Simulator] Starting physiological sensor telemetry streaming...")
    print(f"[Simulator] Target: {endpoint}")
    print(f"[Simulator] Session ID: {session_id} | Mode: {mode} | Interval: {interval}s | Count: {count}\n")

    for i in range(1, count + 1):
        if mode == "elevated" or mode == "stress":
            hr = round(random.uniform(96.0, 120.0), 1)
            spo2 = round(random.uniform(93.0, 96.0), 1)
            temp = round(random.uniform(37.4, 38.2), 2)
        elif mode == "fatigue":
            hr = round(random.uniform(85.0, 102.0), 1)
            spo2 = round(random.uniform(91.0, 94.5), 1)
            temp = round(random.uniform(36.8, 37.3), 2)
        else: # normal
            hr = round(random.uniform(68.0, 78.0), 1)
            spo2 = round(random.uniform(97.5, 99.5), 1)
            temp = round(random.uniform(36.4, 36.8), 2)

        payload = {
            "session_id": session_id,
            "heart_rate": hr,
            "spo2": spo2,
            "temperature": temp,
            "is_simulated": True
        }

        try:
            resp = requests.post(endpoint, json=payload, timeout=5)
            if resp.status_code == 201:
                res_json = resp.json()
                print(f"[{i}/{count}] HR: {hr} bpm | SpO2: {spo2}% | Temp: {temp}°C | Quality: {res_json.get('sensor_quality_score')}% -> SUCCESS")
            else:
                print(f"[{i}/{count}] Error {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"[{i}/{count}] Transmission error: {e}")

        if i < count:
            time.sleep(interval)

    print("\n[Simulator] Telemetry streaming completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESP32 Telemetry Simulator")
    parser.add_argument("--session-id", type=str, required=True, help="Active screening session ID")
    parser.add_argument("--server", type=str, default="http://localhost:5000", help="Backend server URL")
    parser.add_argument("--mode", type=str, choices=["normal", "stress", "fatigue", "elevated"], default="normal", help="Simulation mode")
    parser.add_argument("--interval", type=float, default=1.0, help="Transmission interval in seconds")
    parser.add_argument("--count", type=int, default=5, help="Number of samples to transmit")

    args = parser.parse_args()
    run_simulator(args.session_id, args.server, args.mode, args.interval, args.count)
