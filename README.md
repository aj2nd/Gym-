# AI Gym Head Tracking Backend

This is a FastAPI backend application for an AI Gym Head Tracking system that manages two types of users: members and trainers. It features face recognition-based identification, attendance logging, real-time alerts, and advanced trainer tracking including zone engagement, interaction times, idle detection, and behavior metrics.

---

## Features

- Upload member and trainer data including photos and schedule information
- Generate face embeddings for recognition using a pre-trained face recognition model
- Process CCTV video frames or images to identify members and trainers via facial recognition
- Log attendance with check-in/check-out times and membership status
- Real-time alert system for expired or blacklisted members via WebSockets
- Advanced trainer tracking:
  - Check-in/out times and attendance compliance with schedules
  - Zone tracking within the gym (weights, cardio, personal training rooms)
  - Trainer-member interaction time and simultaneous member handling
  - Idle time detection and productivity measurement
  - Greet detection and facial emotion analysis for behavior and quality metrics
  - Engagement heatmaps and detailed analytics reports

---

## Requirements

- Python 3.8+
- See [`requirements.txt`](./requirements.txt) for Python package dependencies

---

## Installation

1. Clone this repository or download the source code.

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
