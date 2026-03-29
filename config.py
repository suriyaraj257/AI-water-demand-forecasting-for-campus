"""
Configuration module for Smart Campus Water Demand Prediction System.
Loads environment variables and provides system-wide constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ─── Project Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SCRIPTS_DIR = BASE_DIR / "scripts"
DATABASE_DIR = BASE_DIR / "database"

# ─── Database ────────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://user:password@localhost:5432/campus_water_db"
)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME", "campus_water_db")
DB_USER = os.getenv("DB_USER", "user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")

# ─── API ─────────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# ─── External APIs ──────────────────────────────────────────────────────────────
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
WEATHER_CITY = os.getenv("WEATHER_CITY", "Hyderabad")

# ─── Telegram ────────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ─── Model Paths ─────────────────────────────────────────────────────────────────
MODEL_PATH = MODELS_DIR / "water_forecast_model.pkl"
LSTM_MODEL_PATH = MODELS_DIR / "water_forecast_lstm.h5"
SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"

# ─── Alert Thresholds ───────────────────────────────────────────────────────────
TANK_LOW_THRESHOLD = float(os.getenv("TANK_LOW_THRESHOLD", 20))
TANK_HIGH_THRESHOLD = float(os.getenv("TANK_HIGH_THRESHOLD", 95))
SPIKE_DETECTION_MULTIPLIER = float(os.getenv("SPIKE_DETECTION_MULTIPLIER", 2.0))

# ─── Building Types ─────────────────────────────────────────────────────────────
BUILDING_TYPES = ["hostel", "canteen", "academic", "garden"]

BUILDINGS = {
    "BLD-001": {"name": "Main Hostel A", "type": "hostel", "capacity": 500},
    "BLD-002": {"name": "Main Hostel B", "type": "hostel", "capacity": 450},
    "BLD-003": {"name": "Central Canteen", "type": "canteen", "capacity": 300},
    "BLD-004": {"name": "Mess Hall", "type": "canteen", "capacity": 200},
    "BLD-005": {"name": "Engineering Block", "type": "academic", "capacity": 800},
    "BLD-006": {"name": "Science Block", "type": "academic", "capacity": 600},
    "BLD-007": {"name": "Library Complex", "type": "academic", "capacity": 400},
    "BLD-008": {"name": "Central Garden", "type": "garden", "capacity": 0},
    "BLD-009": {"name": "Sports Complex Garden", "type": "garden", "capacity": 0},
    "BLD-010": {"name": "Admin Block", "type": "academic", "capacity": 200},
}

# ─── Tank & Pump Configuration ──────────────────────────────────────────────────
TANKS = {
    "TANK-01": {"name": "Main Overhead Tank", "capacity_liters": 50000, "location": "Central"},
    "TANK-02": {"name": "Hostel Tank", "capacity_liters": 30000, "location": "Hostel Zone"},
    "TANK-03": {"name": "Academic Tank", "capacity_liters": 25000, "location": "Academic Zone"},
}

PUMPS = {
    "PUMP-01": {"name": "Main Pump 1", "flow_rate_lph": 5000, "power_kw": 7.5, "tank": "TANK-01"},
    "PUMP-02": {"name": "Main Pump 2", "flow_rate_lph": 4000, "power_kw": 5.5, "tank": "TANK-01"},
    "PUMP-03": {"name": "Hostel Pump", "flow_rate_lph": 3000, "power_kw": 4.0, "tank": "TANK-02"},
    "PUMP-04": {"name": "Academic Pump", "flow_rate_lph": 3500, "power_kw": 4.5, "tank": "TANK-03"},
}

# ─── Electricity Peak Hours ──────────────────────────────────────────────────────
PEAK_HOURS = list(range(9, 12)) + list(range(18, 22))  # 9-12 AM, 6-10 PM
OFF_PEAK_HOURS = [h for h in range(24) if h not in PEAK_HOURS]
PEAK_RATE_MULTIPLIER = 1.8  # Electricity cost multiplier during peak

# ─── Feature Engineering ─────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "hour_of_day", "day_of_week", "weekend_flag", "is_holiday",
    "temperature", "humidity", "rainfall",
    "temperature_index", "occupancy_ratio",
    "previous_day_usage", "rolling_7_day_average", "rolling_24_hour_usage",
    "building_type_encoded"
]

TARGET_COLUMN = "water_consumption_liters"
