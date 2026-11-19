# 🚀 Real-Time DC Motor Health Monitoring & RUL Prediction System

Sistem pemantauan kesehatan Motor DC berbasis **IoT** dan **Deep Learning (LSTM)** secara *real-time*. Data sensor ditangkap oleh ESP32, dikirim via HTTP ke Node-RED, disimpan di InfluxDB, dan diproses oleh model LSTM untuk memprediksi *Remaining Useful Life* (RUL).

---

## 🛠️ Tech Stack & Architecture

- **Hardware:** ESP32, MPU6050 (Vibration), ADS1115 + Sensor Arus, DS18B20 (Suhu), Hall Effect (RPM).
- **Edge Integration:** Node-RED.
- **Database:** InfluxDB v2 (Time-Series DB).
- **Machine Learning:** Python, TensorFlow / Keras (LSTM Network).
- **Visualization:** Grafana & InfluxDB Data Explorer.
- **Containerization:** Docker & Docker Compose.

---

## 🏗️ Architecture Diagram

```
[ ESP32 Sensors ] ──(Wi-Fi / HTTP)──> [ Node-RED ] ──> [ InfluxDB ]
                                                             │
[ Grafana Dashboard ] <── [ InfluxDB ] <── [ Python LSTM ] ──┘
```

---

## 🚀 Quick Start

### 1. Run InfluxDB & Grafana (Docker)
```bash
docker compose up -d
```

### 2. Setup Python Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run RUL Prediction Pipeline
```bash
python3 ml_pipeline/predict_rul.py
```

---

## 🔒 Security Note
Pastikan untuk membuat file `.env` untuk menyimpan `INFLUX_TOKEN` dan tidak melakukan commit token rahasia ke repository publik.# MAC-Motor-10HP
