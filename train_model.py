"""
Model Training Module for Smart Campus Water Demand Prediction System.

Trains XGBoost and optional LSTM models for water demand forecasting.
Evaluates model performance and saves trained models.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from xgboost import XGBRegressor
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODELS_DIR, DATA_DIR, MODEL_PATH, LSTM_MODEL_PATH, SCALER_PATH,
    FEATURE_COLUMNS, TARGET_COLUMN
)
from scripts.data_pipeline import DataPipeline


# ─── Extended Feature Set ────────────────────────────────────────────────────
ML_FEATURES = [
    "hour_of_day", "day_of_week_num", "weekend_flag", "is_holiday",
    "month", "day_of_month",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "month_sin", "month_cos",
    "temperature", "humidity", "rainfall",
    "temperature_index", "heat_index", "is_rainy",
    "occupancy", "occupancy_ratio",
    "prev_hour_usage", "previous_day_usage",
    "rolling_3_hour_usage", "rolling_24_hour_usage", "rolling_7_day_average",
    "rolling_24h_std", "rolling_24h_max", "rolling_24h_min",
    "building_type_encoded",
    "temp_x_occupancy", "weekend_x_hour",
    "pump_status",
]


class WaterDemandModelTrainer:
    """Trains and evaluates water demand forecasting models."""
    
    def __init__(self):
        self.xgb_model: Optional[XGBRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_importance: Optional[Dict] = None
        self.metrics: Dict = {}
        os.makedirs(MODELS_DIR, exist_ok=True)
    
    def prepare_features(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare feature matrix X and target vector y."""
        available_features = [f for f in ML_FEATURES if f in df.columns]
        missing = set(ML_FEATURES) - set(available_features)
        if missing:
            logger.warning(f"Missing features: {missing}")
        
        X = df[available_features].copy()
        y = df[TARGET_COLUMN].copy()
        
        # Fill any remaining NaN values
        X = X.fillna(X.median())
        
        logger.info(f"Feature matrix: {X.shape}, Target: {y.shape}")
        return X, y
    
    def train_xgboost(
        self,
        df: pd.DataFrame,
        test_ratio: float = 0.2,
        tune_hyperparams: bool = False
    ) -> Dict:
        """Train XGBoost regression model."""
        logger.info("=" * 60)
        logger.info("Training XGBoost Model")
        logger.info("=" * 60)
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Chronological train/test split
        split_idx = int(len(X) * (1 - test_ratio))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"Train samples: {len(X_train):,}, Test samples: {len(X_test):,}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # XGBoost parameters
        params = {
            "n_estimators": 500,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "objective": "reg:squarederror",
        }
        
        if tune_hyperparams:
            params = self._tune_hyperparameters(X_train_scaled, y_train, params)
        
        # Train model
        logger.info("Training XGBoost model...")
        self.xgb_model = XGBRegressor(**params)
        
        self.xgb_model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=50
        )
        
        # Predictions
        y_pred_train = self.xgb_model.predict(X_train_scaled)
        y_pred_test = self.xgb_model.predict(X_test_scaled)
        
        # Ensure non-negative predictions
        y_pred_train = np.maximum(y_pred_train, 0)
        y_pred_test = np.maximum(y_pred_test, 0)
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, y_pred_train, "train")
        test_metrics = self._calculate_metrics(y_test, y_pred_test, "test")
        
        self.metrics["xgboost"] = {
            "train": train_metrics,
            "test": test_metrics,
            "params": params,
            "n_features": len(X.columns),
            "feature_names": list(X.columns),
        }
        
        # Feature importance
        importance = self.xgb_model.feature_importances_
        self.feature_importance = dict(
            sorted(
                zip(X.columns, importance),
                key=lambda x: x[1],
                reverse=True
            )
        )
        
        # Print results
        logger.info("\n" + "=" * 60)
        logger.info("XGBoost Model Results")
        logger.info("=" * 60)
        logger.info(f"Train MAE: {train_metrics['mae']:.2f} liters")
        logger.info(f"Test  MAE: {test_metrics['mae']:.2f} liters")
        logger.info(f"Train RMSE: {train_metrics['rmse']:.2f} liters")
        logger.info(f"Test  RMSE: {test_metrics['rmse']:.2f} liters")
        logger.info(f"Train R²: {train_metrics['r2']:.4f}")
        logger.info(f"Test  R²: {test_metrics['r2']:.4f}")
        logger.info(f"\nTop 10 Features:")
        for feat, imp in list(self.feature_importance.items())[:10]:
            logger.info(f"  {feat}: {imp:.4f}")
        
        # Save model
        self._save_xgboost_model()
        
        return self.metrics["xgboost"]
    
    def train_lstm(
        self,
        df: pd.DataFrame,
        sequence_length: int = 24,
        test_ratio: float = 0.2,
        epochs: int = 50,
        batch_size: int = 64
    ) -> Dict:
        """Train LSTM model for time-series forecasting (optional)."""
        try:
            from tensorflow import keras
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        except ImportError:
            logger.warning("TensorFlow not installed. Skipping LSTM training.")
            return {}
        
        logger.info("=" * 60)
        logger.info("Training LSTM Model")
        logger.info("=" * 60)
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Scale
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - sequence_length):
            X_seq.append(X_scaled[i:i + sequence_length])
            y_seq.append(y.iloc[i + sequence_length])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        logger.info(f"Sequence shape: {X_seq.shape}")
        
        # Split
        split_idx = int(len(X_seq) * (1 - test_ratio))
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        
        # Build LSTM model
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(sequence_length, X.shape[1])),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            Dense(32, activation="relu"),
            Dropout(0.1),
            Dense(1)
        ])
        
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
        ]
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        y_pred_test = model.predict(X_test).flatten()
        y_pred_test = np.maximum(y_pred_test, 0)
        
        test_metrics = self._calculate_metrics(y_test, y_pred_test, "test")
        
        self.metrics["lstm"] = {
            "test": test_metrics,
            "sequence_length": sequence_length,
            "epochs_trained": len(history.history["loss"]),
        }
        
        logger.info("\n" + "=" * 60)
        logger.info("LSTM Model Results")
        logger.info("=" * 60)
        logger.info(f"Test MAE: {test_metrics['mae']:.2f} liters")
        logger.info(f"Test RMSE: {test_metrics['rmse']:.2f} liters")
        logger.info(f"Test R²: {test_metrics['r2']:.4f}")
        
        # Save LSTM model
        model.save(str(LSTM_MODEL_PATH))
        logger.info(f"LSTM model saved to: {LSTM_MODEL_PATH}")
        
        return self.metrics["lstm"]
    
    def compare_models(self) -> pd.DataFrame:
        """Compare XGBoost vs LSTM performance."""
        comparison = []
        for model_name, metrics in self.metrics.items():
            if "test" in metrics:
                comparison.append({
                    "Model": model_name.upper(),
                    "MAE": metrics["test"]["mae"],
                    "RMSE": metrics["test"]["rmse"],
                    "R²": metrics["test"]["r2"],
                    "MAPE (%)": metrics["test"].get("mape", None),
                })
        
        comparison_df = pd.DataFrame(comparison)
        logger.info("\n📊 Model Comparison:")
        logger.info(f"\n{comparison_df.to_string(index=False)}")
        
        return comparison_df
    
    # ─── Private Methods ─────────────────────────────────────────────────────
    
    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, split: str
    ) -> Dict:
        """Calculate regression metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (avoid division by zero)
        mask = y_true > 0
        if mask.sum() > 0:
            mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
        else:
            mape = None
        
        return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}
    
    def _tune_hyperparameters(self, X, y, base_params):
        """Basic hyperparameter tuning using TimeSeriesSplit."""
        from sklearn.model_selection import RandomizedSearchCV
        
        param_dist = {
            "n_estimators": [300, 500, 700],
            "max_depth": [5, 7, 9, 11],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "min_child_weight": [3, 5, 7],
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        
        search = RandomizedSearchCV(
            XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1),
            param_dist,
            n_iter=20,
            scoring="neg_mean_absolute_error",
            cv=tscv,
            random_state=42,
            verbose=1
        )
        
        search.fit(X, y)
        logger.info(f"Best params: {search.best_params_}")
        return {**base_params, **search.best_params_}
    
    def _save_xgboost_model(self):
        """Save XGBoost model, scaler, and metadata."""
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Save model
        joblib.dump(self.xgb_model, str(MODEL_PATH))
        logger.info(f"XGBoost model saved to: {MODEL_PATH}")
        
        # Save scaler
        joblib.dump(self.scaler, str(SCALER_PATH))
        logger.info(f"Scaler saved to: {SCALER_PATH}")
        
        # Save feature importance (convert numpy types to native Python)
        importance_path = MODELS_DIR / "feature_importance.json"
        fi_serializable = {
            k: float(v) for k, v in self.feature_importance.items()
        }
        with open(importance_path, "w") as f:
            json.dump(fi_serializable, f, indent=2)
        
        # Save metrics using a custom encoder to handle all numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        metrics_path = MODELS_DIR / "model_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2, cls=NumpyEncoder)
        
        logger.info("Model artifacts saved successfully")


if __name__ == "__main__":
    # Run data pipeline
    pipeline = DataPipeline()
    df = pipeline.run_pipeline()
    
    # Train models
    trainer = WaterDemandModelTrainer()
    
    # Train XGBoost
    xgb_metrics = trainer.train_xgboost(df, test_ratio=0.2)
    
    # Train LSTM (optional)
    try:
        lstm_metrics = trainer.train_lstm(df, sequence_length=24, epochs=30)
        trainer.compare_models()
    except Exception as e:
        logger.warning(f"LSTM training skipped: {e}")
    
    print("\n✅ Model training complete!")
