# 🔴 CRIMEVIS AI
### Real-Time Crime Detection & Intelligent Public Safety System

![Python](https://img.shields.io/badge/Python-3.13-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)
![Flask](https://img.shields.io/badge/Flask-3.1-lightgrey)

## 🎯 Overview
CRIMEVIS AI is the world's first multi-modal intelligent 
public safety system combining computer vision, audio 
detection, and gesture recognition to detect threats 
and automatically alert authorities without human intervention.

## ✨ Features

### 🔴 Crime Detection
- 👤 Person Detection — YOLOv8 real-time human detection
- 🔪 Weapon Detection — Knives, scissors, dangerous objects
- 🥊 Fight Detection — Aggressive movement detection
- ⏱️ Loitering Detection — Suspicious lingering alert

### 🚑 Medical Emergency
- 🫀 Fall Detection — Person falls to ground
- 😵 Collapse Detection — Person falls and face not visible
- 👁️ Unconscious Detection — No face detected 15+ seconds

### 🆘 Distress Detection
- 🤟 Silent SOS Gesture — Hands raised distress signal
- 🎤 Audio Scream Detection — Real-time scream detection
- 👁️ Facial Monitoring — Eyes closed detection

### 📊 Intelligence
- ⚡ Threat Score 1-10
- 📸 Auto Screenshot saving
- 📧 Email alerts with screenshot
- 🌐 Live web dashboard
- 📄 PDF incident reports

## 🛠️ Tech Stack
- Python 3.13
- OpenCV 4.13
- YOLOv8 Ultralytics
- Flask
- PyAudio
- ReportLab

## 🚀 How To Run

### Full System
python3 crimevis_master.py

### Web Dashboard
python3 dashboard.py
Open → http://localhost:8080

### PDF Report
python3 pdf_report.py

## 📊 Threat Score
- Score 1-3 → ALL CLEAR
- Score 4-6 → MEDIUM THREAT
- Score 7-10 → HIGH THREAT → Auto Alert

## 🎯 Use Cases
- Hospital corridor monitoring
- College campus safety
- Mall and airport security
- Bank and ATM surveillance
- Metro and bus stations
- Elderly care homes

## 📁 Project Files
- crimevis_master.py → Complete master system
- main.py → Crime detection
- fall_detection.py → Fall and medical emergency
- gesture_sos.py → Silent SOS detection
- audio_detection.py → Scream detection
- facial_detection.py → Facial monitoring
- dashboard.py → Live web dashboard
- email_alert.py → Email alerts
- pdf_report.py → PDF reports

## 📄 Research Paper
"CRIMEVIS AI: A Multi-Modal Real-Time Public Safety 
Intelligence System Using Computer Vision, Audio 
Analysis and Gesture Recognition"

## 🎓 Built By
Keerthana S — CSE Engineering Student
GitHub: keerthanasivakumar-dev

