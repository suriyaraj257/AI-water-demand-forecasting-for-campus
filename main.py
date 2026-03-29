"""
Smart Campus Water Demand Prediction and Pump Optimization System

Main entry point for running the complete pipeline:
1. Generate synthetic data
2. Run data pipeline
3. Train ML models
4. Generate predictions
5. Optimize pump schedules
6. Run anomaly detection

Usage:
    python main.py --mode full       # Run everything
    python main.py --mode generate   # Generate data only
    python main.py --mode train      # Train models only
    python main.py --mode predict    # Run prediction only
    python main.py --mode simulate   # Run simulation mode
"""

import argparse
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss} | {level: <8} | {message}", level="INFO")
logger.add("logs/campus_water_{time}.log", rotation="10 MB", retention="7 days", level="DEBUG")

# Ensure paths
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
os.makedirs(BASE_DIR / "logs", exist_ok=True)

from config import DATA_DIR, MODELS_DIR, MODEL_PATH


def run_data_generation():
    """Step 1: Generate synthetic dataset."""
    logger.info("=" * 70)
    logger.info("STEP 1: GENERATING SYNTHETIC DATASET")
    logger.info("=" * 70)
    
    from scripts.generate_data import generate_synthetic_dataset, inject_anomalies, save_dataset
    
    df = generate_synthetic_dataset(
        start_date="2023-01-01",
        end_date="2025-12-31",
        seed=42
    )
    df = inject_anomalies(df, anomaly_rate=0.005)
    save_dataset(df, "campus_water_data.csv")
    
    return df


def run_data_pipeline():
    """Step 2: Run data pipeline (clean + feature engineer)."""
    logger.info("=" * 70)
    logger.info("STEP 2: RUNNING DATA PIPELINE")
    logger.info("=" * 70)
    
    from scripts.data_pipeline import DataPipeline
    
    pipeline = DataPipeline()
    df = pipeline.run_pipeline()
    
    return df


def run_model_training(df=None):
    """Step 3: Train ML models."""
    logger.info("=" * 70)
    logger.info("STEP 3: TRAINING ML MODELS")
    logger.info("=" * 70)
    
    if df is None:
        from scripts.data_pipeline import DataPipeline
        pipeline = DataPipeline()
        df = pipeline.run_pipeline()
    
    from train_model import WaterDemandModelTrainer
    
    trainer = WaterDemandModelTrainer()
    
    # Train XGBoost
    xgb_metrics = trainer.train_xgboost(df, test_ratio=0.2)
    
    # Try LSTM
    try:
        lstm_metrics = trainer.train_lstm(df, sequence_length=24, epochs=30)
        comparison = trainer.compare_models()
        logger.info(f"\n{comparison.to_string(index=False)}")
    except Exception as e:
        logger.warning(f"LSTM training skipped: {e}")
    
    return trainer


def run_prediction():
    """Step 4: Generate water demand predictions."""
    logger.info("=" * 70)
    logger.info("STEP 4: GENERATING PREDICTIONS")
    logger.info("=" * 70)
    
    from predict import WaterDemandPredictor, get_default_weather, get_default_occupancy
    
    predictor = WaterDemandPredictor(model_type="xgboost")
    weather = get_default_weather()
    occupancy = get_default_occupancy()
    
    # Next-day forecast
    next_day = predictor.predict_next_day(weather, occupancy)
    
    logger.info(f"\n🔮 Next-Day Forecast:")
    logger.info(f"   Total Campus Demand: {next_day['total_campus_demand_liters']:,.1f} liters")
    for btype, demand in next_day["type_summary"].items():
        logger.info(f"   {btype}: {demand:,.1f} liters")
    
    # Save predictions
    pred_path = DATA_DIR / "latest_predictions.json"
    with open(pred_path, "w") as f:
        json.dump(next_day, f, indent=2, default=str)
    logger.info(f"Predictions saved to: {pred_path}")
    
    return next_day


def run_pump_optimization(predictions=None):
    """Step 5: Optimize pump schedules."""
    logger.info("=" * 70)
    logger.info("STEP 5: OPTIMIZING PUMP SCHEDULE")
    logger.info("=" * 70)
    
    from pump_optimizer import PumpOptimizer
    import numpy as np
    
    optimizer = PumpOptimizer()
    
    if predictions is None:
        predictions = {
            "total_campus_demand_liters": 45000,
            "hourly_demand": [
                {"hour": h, "total_demand_liters": 1500 + 500 * np.sin(np.pi * h / 12)}
                for h in range(24)
            ],
        }
    
    result = optimizer.optimize_schedule(predictions)
    
    # Save schedule
    schedule_dict = optimizer.to_dict(result)
    schedule_path = DATA_DIR / "latest_pump_schedule.json"
    with open(schedule_path, "w") as f:
        json.dump(schedule_dict, f, indent=2, default=str)
    logger.info(f"Pump schedule saved to: {schedule_path}")
    
    return result


def run_anomaly_detection():
    """Step 6: Run anomaly detection."""
    logger.info("=" * 70)
    logger.info("STEP 6: RUNNING ANOMALY DETECTION")
    logger.info("=" * 70)
    
    from scripts.anomaly_detection import AnomalyDetector
    
    detector = AnomalyDetector()
    
    # Test with tank levels
    test_levels = {
        "TANK-01": {"level_percent": 45, "name": "Main Overhead Tank"},
        "TANK-02": {"level_percent": 28, "name": "Hostel Tank"},
        "TANK-03": {"level_percent": 62, "name": "Academic Tank"},
    }
    
    alerts = detector.check_tank_levels(test_levels)
    
    logger.info(f"\n🚨 Active Alerts: {len(alerts)}")
    for alert in alerts:
        logger.info(f"  [{alert.severity.upper()}] {alert.title}")
    
    return detector


def run_simulation():
    """Run simulation mode with random events."""
    logger.info("=" * 70)
    logger.info("SIMULATION MODE")
    logger.info("=" * 70)
    
    from scripts.simulation import SimulationEngine
    from scripts.anomaly_detection import AnomalyDetector
    from pump_optimizer import PumpOptimizer
    
    sim = SimulationEngine(seed=42)
    results = sim.run_full_simulation()
    
    # Run anomaly detection on simulated data
    detector = AnomalyDetector()
    if "tank_anomaly" in results:
        alerts = detector.check_tank_levels(results["tank_anomaly"])
        logger.info(f"\n🚨 Simulation Alerts: {len(alerts)}")
    
    # Re-optimize with spiked demand
    optimizer = PumpOptimizer()
    if "spiked_demand" in results:
        schedule = optimizer.optimize_schedule(results["spiked_demand"])
        logger.info(f"\n📋 Emergency Schedule Generated")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Smart Campus Water Demand Prediction System"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "generate", "pipeline", "train", "predict", "optimize", "detect", "simulate"],
        default="full",
        help="Execution mode"
    )
    args = parser.parse_args()
    
    logger.info("🌊 Smart Campus Water Demand Prediction System")
    logger.info(f"   Mode: {args.mode}")
    logger.info(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    if args.mode == "generate":
        run_data_generation()
    
    elif args.mode == "pipeline":
        run_data_pipeline()
    
    elif args.mode == "train":
        run_model_training()
    
    elif args.mode == "predict":
        run_prediction()
    
    elif args.mode == "optimize":
        run_pump_optimization()
    
    elif args.mode == "detect":
        run_anomaly_detection()
    
    elif args.mode == "simulate":
        run_simulation()
    
    elif args.mode == "full":
        # Run everything
        df = run_data_generation()
        df = run_data_pipeline()
        trainer = run_model_training(df)
        predictions = run_prediction()
        schedule = run_pump_optimization(predictions)
        detector = run_anomaly_detection()
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ FULL PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"   Data: {DATA_DIR}")
        logger.info(f"   Models: {MODELS_DIR}")
        logger.info(f"   Predictions: {DATA_DIR / 'latest_predictions.json'}")
        logger.info(f"   Pump Schedule: {DATA_DIR / 'latest_pump_schedule.json'}")
    
    logger.info("\n🏁 Execution complete!")


if __name__ == "__main__":
    main()
