# 🌊 Smart Campus Water Demand Prediction & Pump Optimization System

An AI-powered system that forecasts next-day and next-week water demand for a university campus and recommends optimal pump schedules to reduce electricity usage and prevent water shortages or overflow.

---

## 🎯 Features

- **📊 ML-Based Forecasting** — XGBoost + optional LSTM models predict hourly water demand per building
- **⚡ Pump Optimization** — Cost-minimizing pump scheduling that avoids peak electricity hours
- **🚨 Anomaly Detection** — Real-time leak detection, spike alerts, and pump malfunction monitoring
- **📈 Interactive Dashboard** — 5-page Streamlit dashboard with Plotly visualizations
- **🔗 REST API** — FastAPI backend with prediction, scheduling, and dashboard endpoints
- **🤖 Automation** — n8n workflow + APScheduler for daily forecasting and reporting
- **🎮 Simulation Mode** — Stress-test the system with random demand spikes and failures

---

## 🏗️ Architecture

```
campus-water-ai/
├── api/                    # FastAPI REST API
│   ├── __init__.py
│   └── app.py              # API endpoints
├── dashboard/              # Streamlit Dashboard
│   ├── __init__.py
│   └── app.py              # 5-page interactive dashboard
├── data/                   # Data storage
│   ├── campus_water_data.csv
│   ├── processed_water_data.csv
│   ├── latest_predictions.json
│   └── latest_pump_schedule.json
├── database/               # Database schema
│   └── schema.sql          # PostgreSQL schema
├── models/                 # Trained ML models
│   ├── water_forecast_model.pkl
│   ├── water_forecast_lstm.h5
│   ├── feature_scaler.pkl
│   ├── feature_importance.json
│   └── model_metrics.json
├── notebooks/              # Jupyter notebooks
│   └── exploration.ipynb
├── scripts/                # Utility scripts
│   ├── generate_data.py    # Synthetic data generator
│   ├── data_pipeline.py    # ETL pipeline
│   ├── anomaly_detection.py
│   ├── simulation.py       # Simulation engine
│   ├── scheduler.py        # Automated scheduler
│   └── n8n_workflow.json   # n8n automation
├── config.py               # System configuration
├── main.py                 # Main entry point
├── train_model.py          # Model training
├── predict.py              # Prediction engine
├── pump_optimizer.py       # Pump schedule optimizer
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
└── README.md               # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd campus-water-ai
pip install -r requirements.txt
```

### 2. Generate Data & Train Model

```bash
# Full pipeline: generate data → process → train → predict → optimize
python main.py --mode full
```

Or run individual steps:

```bash
python main.py --mode generate    # Generate synthetic data
python main.py --mode pipeline    # Clean & feature engineer
python main.py --mode train       # Train XGBoost + LSTM
python main.py --mode predict     # Generate forecasts
python main.py --mode optimize    # Optimize pump schedule
python main.py --mode simulate    # Run simulation mode
```

### 3. Start API Server

```bash
cd campus-water-ai
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

API Docs available at: `http://localhost:8000/docs`

### 4. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard available at: `http://localhost:8501`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Generate water demand forecast |
| `GET` | `/predict/next-day` | Quick next-day prediction |
| `GET` | `/predict/next-week` | Quick next-week prediction |
| `GET` | `/pump-schedule` | Get optimized pump schedule |
| `POST` | `/pump-schedule/from-prediction` | Predict + optimize in one call |
| `GET` | `/dashboard-data` | Full analytics data |
| `GET` | `/dashboard-data/tanks` | Current tank status |
| `GET` | `/dashboard-data/buildings` | Building information |
| `GET` | `/alerts` | Active system alerts |
| `POST` | `/simulate` | Run simulation scenario |
| `POST` | `/simulate/full` | Full stress test |

### Example: Predict Demand

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "weather": {"temperature": 35, "humidity": 50, "rainfall": 0},
    "is_holiday": false,
    "prediction_hours": 24
  }'
```

---

## 📊 Dashboard Pages

1. **🏫 Campus Water Overview** — Tank gauges, building consumption, system status
2. **🔮 Demand Forecast** — Hourly demand charts, building heatmap, type breakdown
3. **⚡ Pump Schedule** — Optimized timings, cost analysis, tank projections
4. **📊 Historical Analysis** — Consumption trends, day-of-week patterns, anomalies
5. **🎯 Model Accuracy** — MAE/RMSE/R², feature importance, model comparison

---

## 🤖 Automation

### n8n Workflow

Import `scripts/n8n_workflow.json` into n8n. The workflow:

1. Fetches weather forecast (OpenWeatherMap)
2. Fetches current occupancy data
3. Runs prediction API
4. Generates pump schedule
5. Stores results in PostgreSQL
6. Sends daily report to Telegram

### APScheduler

```bash
# Run scheduler daemon
python scripts/scheduler.py --daemon

# Run prediction immediately
python scripts/scheduler.py --run-now
```

---

## 🗄️ Database Setup

```bash
# Create database
createdb campus_water_db

# Run schema
psql -d campus_water_db -f database/schema.sql
```

---

## 🔧 Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key settings:
- `DATABASE_URL` — PostgreSQL connection string
- `WEATHER_API_KEY` — OpenWeatherMap API key
- `TELEGRAM_BOT_TOKEN` — For automated reports
- `TANK_LOW_THRESHOLD` — Alert threshold (default: 20%)

---

## 📈 Model Details

### XGBoost Features

- **Temporal**: hour, day, weekend, holiday, cyclical encodings
- **Weather**: temperature, humidity, rainfall, heat index
- **Usage**: previous hour, rolling 3h/24h/7d averages
- **Building**: type encoding, occupancy ratio
- **Interactions**: temp × occupancy, weekend × hour

### Performance (Typical)

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| XGBoost | ~42 L | ~68 L | ~0.92 |
| LSTM | ~49 L | ~73 L | ~0.90 |

---

## 🎮 Simulation Mode

Test system robustness with:

```bash
python main.py --mode simulate
```

Simulates:
- **Demand spikes** (2-5× normal) on random buildings
- **Extreme weather** (heatwave, heavy rain, cold snap)
- **Pump failures** (complete/partial/intermittent)
- **Tank anomalies** (critically low/overflow risk)

---

## 📝 License

MIT License — For educational and campus operations use.
