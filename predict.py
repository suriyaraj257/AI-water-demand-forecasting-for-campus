"""
Water Demand Prediction Engine for Smart Campus System.

Loads trained models and generates forecasts for next-day and next-week demand.
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    MODEL_PATH, LSTM_MODEL_PATH, SCALER_PATH, MODELS_DIR,
    BUILDINGS, BUILDING_TYPES
)
from train_model import ML_FEATURES


class WaterDemandPredictor:
    """Predicts water demand using trained models."""
    
    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_names = ML_FEATURES
        self._load_model()
    
    def _load_model(self):
        """Load trained model and scaler from disk."""
        try:
            if self.model_type == "xgboost":
                self.model = joblib.load(str(MODEL_PATH))
                logger.info(f"XGBoost model loaded from {MODEL_PATH}")
            elif self.model_type == "lstm":
                from tensorflow.keras.models import load_model
                self.model = load_model(str(LSTM_MODEL_PATH))
                logger.info(f"LSTM model loaded from {LSTM_MODEL_PATH}")
            
            if SCALER_PATH.exists():
                self.scaler = joblib.load(str(SCALER_PATH))
                logger.info("Feature scaler loaded")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict_water_demand(
        self,
        next_day_weather: Dict,
        occupancy: Dict,
        schedule: Dict = None,
        prediction_hours: int = 24
    ) -> Dict:
        """
        Predict water demand for upcoming period.
        
        Args:
            next_day_weather: Dict with keys 'temperature', 'humidity', 'rainfall'
                             Can be hourly lists or single values.
            occupancy: Dict with building_id -> expected_occupancy mappings
            schedule: Dict with 'is_holiday' and other schedule info
            prediction_hours: Number of hours to predict (24 for next day, 168 for next week)
        
        Returns:
            Dict with per-building and total campus predictions
        """
        logger.info(f"Generating {prediction_hours}-hour forecast...")
        
        now = datetime.now()
        predictions = {}
        total_demand = 0
        hourly_demand = []
        
        for hour_offset in range(prediction_hours):
            target_time = now + timedelta(hours=hour_offset)
            hour = target_time.hour
            day_of_week = target_time.weekday()
            month = target_time.month
            day_of_month = target_time.day
            
            # Get weather for this hour
            if isinstance(next_day_weather.get("temperature"), list):
                idx = hour_offset % len(next_day_weather["temperature"])
                temp = next_day_weather["temperature"][idx]
                humidity = next_day_weather["humidity"][idx]
                rainfall = next_day_weather["rainfall"][idx]
            else:
                temp = next_day_weather.get("temperature", 30)
                humidity = next_day_weather.get("humidity", 60)
                rainfall = next_day_weather.get("rainfall", 0)
            
            is_holiday = schedule.get("is_holiday", 0) if schedule else 0
            is_weekend = 1 if day_of_week >= 5 else 0
            
            hour_total = 0
            
            for building_id, building_info in BUILDINGS.items():
                bld_occupancy = occupancy.get(building_id, building_info["capacity"] * 0.5)
                capacity = max(building_info["capacity"], 1)
                
                type_map = {"hostel": 0, "canteen": 1, "academic": 2, "garden": 3}
                
                # Build feature vector
                features = {
                    "hour_of_day": hour,
                    "day_of_week_num": day_of_week,
                    "weekend_flag": is_weekend,
                    "is_holiday": is_holiday,
                    "month": month,
                    "day_of_month": day_of_month,
                    "hour_sin": np.sin(2 * np.pi * hour / 24),
                    "hour_cos": np.cos(2 * np.pi * hour / 24),
                    "day_sin": np.sin(2 * np.pi * day_of_week / 7),
                    "day_cos": np.cos(2 * np.pi * day_of_week / 7),
                    "month_sin": np.sin(2 * np.pi * month / 12),
                    "month_cos": np.cos(2 * np.pi * month / 12),
                    "temperature": temp,
                    "humidity": humidity,
                    "rainfall": rainfall,
                    "temperature_index": temp * 0.6 + (100 - humidity) * 0.4,
                    "heat_index": temp + 0.5 * humidity if temp > 30 else temp,
                    "is_rainy": 1 if rainfall > 0 else 0,
                    "occupancy": bld_occupancy,
                    "occupancy_ratio": bld_occupancy / capacity,
                    "prev_hour_usage": 0,  # Will use rolling average
                    "previous_day_usage": 0,
                    "rolling_3_hour_usage": 0,
                    "rolling_24_hour_usage": 0,
                    "rolling_7_day_average": 0,
                    "rolling_24h_std": 0,
                    "rolling_24h_max": 0,
                    "rolling_24h_min": 0,
                    "building_type_encoded": type_map.get(building_info["type"], 2),
                    "temp_x_occupancy": (temp * 0.6 + (100 - humidity) * 0.4) * (bld_occupancy / capacity),
                    "weekend_x_hour": is_weekend * hour,
                    "pump_status": 1 if hour in [4, 5, 6, 21, 22] else 0,
                }
                
                # Update lag features from previous predictions
                if building_id in predictions and len(predictions[building_id]) > 0:
                    prev_preds = [p["predicted_liters"] for p in predictions[building_id]]
                    features["prev_hour_usage"] = prev_preds[-1] if prev_preds else 0
                    features["rolling_3_hour_usage"] = np.mean(prev_preds[-3:]) if len(prev_preds) >= 3 else np.mean(prev_preds) if prev_preds else 0
                    features["rolling_24_hour_usage"] = np.mean(prev_preds[-24:]) if len(prev_preds) >= 24 else np.mean(prev_preds) if prev_preds else 0
                    if len(prev_preds) >= 24:
                        features["previous_day_usage"] = prev_preds[-24]
                        features["rolling_24h_std"] = np.std(prev_preds[-24:])
                        features["rolling_24h_max"] = np.max(prev_preds[-24:])
                        features["rolling_24h_min"] = np.min(prev_preds[-24:])
                
                # Create feature DataFrame
                feature_df = pd.DataFrame([features])
                
                # Ensure correct column order
                available_features = [f for f in self.feature_names if f in feature_df.columns]
                feature_df = feature_df[available_features]
                
                # Scale features
                if self.scaler:
                    feature_scaled = pd.DataFrame(
                        self.scaler.transform(feature_df),
                        columns=feature_df.columns
                    )
                else:
                    feature_scaled = feature_df
                
                # Predict
                pred = self.model.predict(feature_scaled)[0]
                pred = max(0, float(pred))
                
                # Store prediction
                if building_id not in predictions:
                    predictions[building_id] = []
                
                predictions[building_id].append({
                    "datetime": target_time.isoformat(),
                    "hour": hour,
                    "predicted_liters": round(pred, 1),
                    "building_name": building_info["name"],
                    "building_type": building_info["type"],
                })
                
                hour_total += pred
            
            hourly_demand.append({
                "datetime": target_time.isoformat(),
                "hour": hour,
                "total_demand_liters": round(hour_total, 1),
            })
            total_demand += hour_total
        
        # ── Build response ───────────────────────────────────────────────────
        building_summaries = {}
        for bld_id, preds in predictions.items():
            total = sum(p["predicted_liters"] for p in preds)
            peak = max(preds, key=lambda x: x["predicted_liters"])
            building_summaries[bld_id] = {
                "building_name": BUILDINGS[bld_id]["name"],
                "building_type": BUILDINGS[bld_id]["type"],
                "total_predicted_liters": round(total, 1),
                "peak_hour": peak["hour"],
                "peak_consumption": round(peak["predicted_liters"], 1),
                "avg_hourly": round(total / prediction_hours, 1),
                "hourly_predictions": preds,
            }
        
        # Group by building type
        type_summary = {}
        for bld_id, summary in building_summaries.items():
            btype = summary["building_type"]
            if btype not in type_summary:
                type_summary[btype] = 0
            type_summary[btype] += summary["total_predicted_liters"]
        
        result = {
            "prediction_generated_at": now.isoformat(),
            "prediction_hours": prediction_hours,
            "total_campus_demand_liters": round(total_demand, 1),
            "hourly_demand": hourly_demand,
            "building_predictions": building_summaries,
            "type_summary": {k: round(v, 1) for k, v in type_summary.items()},
            "model_type": self.model_type,
        }
        
        logger.info(f"Forecast complete: {round(total_demand, 1)} liters total demand")
        return result
    
    def predict_next_day(
        self, weather: Dict, occupancy: Dict, schedule: Dict = None
    ) -> Dict:
        """Convenience method for next-day prediction."""
        return self.predict_water_demand(weather, occupancy, schedule, prediction_hours=24)
    
    def predict_next_week(
        self, weather: Dict, occupancy: Dict, schedule: Dict = None
    ) -> Dict:
        """Convenience method for next-week prediction."""
        return self.predict_water_demand(weather, occupancy, schedule, prediction_hours=168)


def get_default_weather() -> Dict:
    """Return default weather forecast for testing."""
    hours = list(range(24))
    return {
        "temperature": [22 + 8 * np.sin(np.pi * (h - 6) / 12) for h in hours],
        "humidity": [70 - 15 * np.sin(np.pi * (h - 6) / 12) for h in hours],
        "rainfall": [0] * 24,
    }


def get_default_occupancy() -> Dict:
    """Return default occupancy for testing."""
    return {
        bld_id: int(info["capacity"] * 0.6)
        for bld_id, info in BUILDINGS.items()
    }


if __name__ == "__main__":
    predictor = WaterDemandPredictor(model_type="xgboost")
    
    weather = get_default_weather()
    occupancy = get_default_occupancy()
    
    # Next-day forecast
    result = predictor.predict_next_day(weather, occupancy)
    
    print("\n🔮 Next-Day Water Demand Forecast")
    print("=" * 60)
    print(f"Total Campus Demand: {result['total_campus_demand_liters']:,.1f} liters")
    print(f"\nBy Building Type:")
    for btype, demand in result["type_summary"].items():
        print(f"  {btype}: {demand:,.1f} liters")
    print(f"\nPer Building:")
    for bld_id, summary in result["building_predictions"].items():
        print(f"  {summary['building_name']}: {summary['total_predicted_liters']:,.1f}L (peak at {summary['peak_hour']}:00)")
