import os
import json
import pandas as pd
import numpy as np
import pickle
import time
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import uvicorn
from fastapi import FastAPI, Request, Response, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import schedule
import random
import threading
import csv
from io import StringIO
from statsmodels.tsa.statespace.sarimax import SARIMAX
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("main.log"), logging.StreamHandler()]
)
logger = logging.getLogger("Main")

# FastAPI setup
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Constants
FAILURE_TYPES = ['General Failure', 'Excessive Tool Wear', 'Overheating', 'Vibration Issue']
ACTIONS = [
    'Examine machine alignment and spindle condition',
    'Inspect cooling system and check lubrication',
    'No action required',
    'Replace tool and adjust feed rate',
    'Schedule maintenance check immediately'
]
IMAGE_PATHS = {
    "Excessive Tool Wear": "images/wear and tear.png",
    "General Failure": "images/Predictive-maintenance.png",
    "Vibration Issue": "images/vibration.png",
    "Overheating": "images/overheating.png",
    "default": "images/default.png"
}

# Load model and recommendation system
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
        logger.info(f"Model feature names: {model.feature_names_in_ if hasattr(model, 'feature_names_in_') else 'Unknown'}")
    with open('recommendation_system.pkl', 'rb') as f:
        recommendation_system = pickle.load(f)
        df = recommendation_system['df']
        scaler = recommendation_system['scaler']
        X_scaled = recommendation_system['X_scaled']
except FileNotFoundError as e:
    logger.error(f"Failed to load model or recommendation system: {e}")
    raise

# Global variables
machines = {"Machine1": None, "Machine2": None}
machine_histories = {"Machine1": [], "Machine2": []}  # Unified history storage
simulated_failure = None
settings = {"alert_thresholds": {"VibrationLevel": 8.0}}

class RandomDataGenerator:
    def __init__(self):
        self.parameter_ranges = {
            'OperationTime': (1, 2000), 'CuttingSpeed': (50, 500), 'FeedRate': (0.1, 2.0),
            'ToolDiameter': (5, 50), 'SpindleSpeed': (1000, 20000), 'MotorCurrent': (1, 100),
            'PowerConsumption': (10, 1000), 'LubricationLevel': (0, 100),
            'CoolingSystemEfficiency': (0, 100), 'CycleTime': (1, 60), 'MachineAge': (0, 20),
            'PartDefectRate': (0, 10), 'MaintenanceFrequency': (1, 365), 'VibrationLevel': (0, 10),
            'MachineHealthStatus': (0, 100), 'ToolWearRate': (0, 1), 'CuttingTemperature': (20, 500),
            'MaterialHardness': (50, 500), 'PressureLevel': (0, 1000), 'MachineStability': (0, 100),
            'PowerSurgeRate': (0, 5), 'MachineCalibration': (0, 100), 'ToolChangeInterval': (1, 1000),
            'SpindleHealthStatus': (0, 100), 'CoolantFlowRate': (0, 100), 'CuttingForce': (0, 1000),
            'MachineLoad': (0, 100), 'SpindleTorque': (0, 500), 'PowerSupplyVoltage': (200, 240)
        }
        self.categorical_params = {
            'ToolType': ['HSS', 'Carbide', 'Ceramic'],
            'MaterialType': ['Aluminum', 'Steel', 'Titanium'],
            'MachineAlignmentStatus': ['Aligned', 'Mis-Aligned'],
            'OperationMode': ['Manual', 'Semi-Automatic', 'Automatic'],
            'SpindleCondition': ['Good', 'Warning']
        }
        self.failure_scenarios = {
            "Overheating": {"CuttingTemperature": (400, 500), "CoolingSystemEfficiency": (0, 20)},
            "Excessive Tool Wear": {"ToolWearRate": (0.8, 1.0), "CuttingSpeed": (400, 500)},
            "Vibration Issue": {"VibrationLevel": (8, 10), "MachineStability": (0, 20)},
            "General Failure": {"MachineHealthStatus": (0, 20), "MaintenanceFrequency": (300, 365)}
        }
        self.time_series_data = {}  # Store time-series per machine

    def generate_single_record(self, failure_type: Optional[str] = None, machine_id: str = "Machine1") -> Dict[str, Any]:
        record = {}
        for param, (min_val, max_val) in self.parameter_ranges.items():
            record[param] = round(random.uniform(min_val, max_val), 2)
        for param, options in self.categorical_params.items():
            record[param] = random.choice(options)
        if failure_type and failure_type in self.failure_scenarios:
            for param, (min_val, max_val) in self.failure_scenarios[failure_type].items():
                record[param] = round(random.uniform(min_val, max_val), 2)
        elif random.random() < 0.2:
            failure_type = random.choice(FAILURE_TYPES)
            for param, (min_val, max_val) in self.failure_scenarios[failure_type].items():
                record[param] = round(random.uniform(min_val, max_val), 2)
        
        cutting_speed = record.get('CuttingSpeed', 275)
        tool_wear = record.get('ToolWearRate', 0.5)
        vibration = record.get('VibrationLevel', 5)
        record['PerformanceScore'] = round(max(0, min(100, 100 - (tool_wear * 30) - (vibration * 5) + (cutting_speed / 10))), 2)

        record["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record["MachineID"] = machine_id
        
        if machine_id not in self.time_series_data:
            self.time_series_data[machine_id] = []
        self.time_series_data[machine_id].append(record)
        if len(self.time_series_data[machine_id]) > 50:
            self.time_series_data[machine_id].pop(0)
        
        return record

    def generate_batch(self, size: int, failure_distribution: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        if failure_distribution is None:
            failure_distribution = {"No Failures Detected": 0.8}
            total_failure_prob = 0.2 / len(FAILURE_TYPES)
            for failure in FAILURE_TYPES:
                failure_distribution[failure] = total_failure_prob
        
        batch = []
        for _ in range(size):
            failure_type = random.choices(list(failure_distribution.keys()), weights=list(failure_distribution.values()))[0]
            record = self.generate_single_record(failure_type if failure_type != "No Failures Detected" else None)
            batch.append(record)
        return batch

class BatchProcessor:
    def __init__(self):
        self.model = model
        self.recommendation_system = recommendation_system
        self.df = df
        self.scaler = scaler
        self.X_scaled = X_scaled
        self.FAILURE_TYPES = FAILURE_TYPES
        self.ACTIONS = ACTIONS
        self.generator = RandomDataGenerator()
        os.makedirs("processed_data", exist_ok=True)
        os.makedirs("predictions", exist_ok=True)
        logger.info("BatchProcessor initialized")

    def process_single_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        features = self._prepare_features(record)
        prediction = self.model.predict(features)
        logger.info(f"Raw model prediction: {prediction.tolist()}")
        predicted_failure, recommendation_text, actions = self._map_prediction(prediction[0])
        if record.get("VibrationLevel", 0) > settings["alert_thresholds"]["VibrationLevel"]:
            predicted_failure = "Vibration Issue"
            recommendation_text = "Take action for Vibration Issue"
            actions = ["Schedule maintenance check immediately"]
        recommendations = self._get_recommendations(predicted_failure) if predicted_failure != "No Failures Detected" else []
        health_score = self._calculate_health_score(record)
        forecast = self.forecast_performance(record["MachineID"])
        return {
            "record": record,
            "prediction": {
                "failure_type": predicted_failure,
                "recommendation_text": recommendation_text,
                "actions": actions,
                "similar_failures": recommendations,
                "health_score": health_score,
                "forecast": forecast,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }

    def _prepare_features(self, record: Dict[str, Any]) -> pd.DataFrame:
        alignment_map = {'Aligned': 0, 'Mis-Aligned': 1}
        mode_map = {'Manual': 2, 'Semi-Automatic': 1, 'Automatic': 0}
        spindle_map = {'Good': 0, 'Warning': 1}
        
        features = [
            record.get('OperationTime', 0), record.get('CuttingSpeed', 0), record.get('FeedRate', 0),
            record.get('ToolDiameter', 0), record.get('SpindleSpeed', 0), record.get('MotorCurrent', 0),
            record.get('PowerConsumption', 0), record.get('LubricationLevel', 0),
            record.get('CoolingSystemEfficiency', 0), record.get('CycleTime', 0),
            record.get('MachineAge', 0), record.get('PartDefectRate', 0),
            record.get('MaintenanceFrequency', 0), record.get('VibrationLevel', 0),
            record.get('MachineHealthStatus', 0), record.get('ToolWearRate', 0),
            record.get('CuttingTemperature', 0), record.get('MaterialHardness', 0),
            record.get('PressureLevel', 0), record.get('MachineStability', 0),
            alignment_map.get(record.get('MachineAlignmentStatus', 'Aligned'), 0),
            record.get('PowerSurgeRate', 0), record.get('MachineCalibration', 0),
            record.get('ToolChangeInterval', 0), record.get('SpindleHealthStatus', 0),
            record.get('CoolantFlowRate', 0), record.get('CuttingForce', 0),
            mode_map.get(record.get('OperationMode', 'Automatic'), 0),
            record.get('MachineLoad', 0), record.get('SpindleTorque', 0),
            spindle_map.get(record.get('SpindleCondition', 'Good'), 0),
            record.get('PowerSupplyVoltage', 0),
            1 if record.get('ToolType', 'HSS') == 'Carbide' else 0,
            1 if record.get('ToolType', 'HSS') == 'Ceramic' else 0,
            1 if record.get('ToolType', 'HSS') == 'HSS' else 0,
            1 if record.get('MaterialType', 'Steel') == 'Aluminum' else 0,
            1 if record.get('MaterialType', 'Steel') == 'Steel' else 0,
            1 if record.get('MaterialType', 'Steel') == 'Titanium' else 0
        ]
        
        feature_names = [
            'OperationTime', 'CuttingSpeed', 'FeedRate', 'ToolDiameter', 'SpindleSpeed', 'MotorCurrent',
            'PowerConsumption', 'LubricationLevel', 'CoolingSystemEfficiency', 'CycleTime', 'MachineAge',
            'PartDefectRate', 'MaintenanceFrequency', 'VibrationLevel', 'MachineHealthStatus', 'ToolWearRate',
            'CuttingTemperature', 'MaterialHardness', 'PressureLevel', 'MachineStability',
            'MachineAlignmentStatus', 'PowerSurgeRate', 'MachineCalibration', 'ToolChangeInterval',
            'SpindleHealthStatus', 'CoolantFlowRate', 'CuttingForce', 'OperationMode', 'MachineLoad',
            'SpindleTorque', 'SpindleCondition', 'PowerSupplyVoltage', 'Carbide', 'Ceramic', 'HSS',
            'Aluminum', 'Steel', 'Titanium'
        ]
        return pd.DataFrame([features], columns=feature_names)

    def _map_prediction(self, output: np.ndarray) -> Tuple[str, str, List[str]]:
        output_list = output.tolist()
        if sum(output_list) == 0:
            return "No Failures Detected", "No action required", ["No action required"]
        failure_idx = output_list.index(1) if 1 in output_list else 0
        failure = self.FAILURE_TYPES[failure_idx % len(self.FAILURE_TYPES)]
        action = self.ACTIONS[failure_idx % len(self.ACTIONS)]
        return failure, f"Take action for {failure}", [action]

    def _get_recommendations(self, predicted_failure: str) -> List[Dict[str, str]]:
        if predicted_failure in self.FAILURE_TYPES:
            image_path = IMAGE_PATHS.get(predicted_failure, IMAGE_PATHS["default"])
            return [{"name": f"Related to {predicted_failure}", "description": "Check similar issues", "image": image_path}]
        return []

    def _calculate_health_score(self, record: Dict[str, Any]) -> Dict[str, Any]:
        health = record.get('MachineHealthStatus', 50)
        wear = record.get('ToolWearRate', 0.5)
        temp = record.get('CuttingTemperature', 250)
        vibe = record.get('VibrationLevel', 5)
        score = max(0, min(100, int(health - (wear * 50) - (temp / 10) - (vibe * 5))))
        status = "Good" if score > 70 else "Warning" if score > 30 else "Critical"
        return {"score": score, "status": status}

    def forecast_performance(self, machine_id: str) -> Dict[str, Any]:
        if machine_id not in self.generator.time_series_data or len(self.generator.time_series_data[machine_id]) < 10:
            return {
                "health_trend": "Insufficient data",
                "health_forecast": [],
                "performance_trend": "Insufficient data",
                "performance_forecast": [],
                "confidence": 0
            }
        
        ts_data = self.generator.time_series_data[machine_id]
        df_ts = pd.DataFrame(ts_data)
        df_ts["timestamp"] = pd.to_datetime(df_ts["timestamp"])
        df_ts = df_ts.set_index("timestamp")
        
        health_target = df_ts["MachineHealthStatus"]
        health_exog = df_ts[["VibrationLevel", "ToolWearRate", "CuttingTemperature"]]
        try:
            health_model = SARIMAX(health_target, exog=health_exog, order=(1, 1, 1))
            health_fit = health_model.fit(disp=False)
            health_forecast = health_fit.forecast(steps=5, exog=health_exog[-5:])
            health_avg = round(health_forecast.mean(), 2)
            health_trend = "Improving" if health_avg > health_target.iloc[-1] else "Declining"
            health_confidence = round(1 - health_fit.pvalues.mean(), 2)
            health_forecast_list = [round(val, 2) for val in health_forecast.tolist()]
        except Exception as e:
            logger.error(f"Health forecast error: {e}")
            health_trend, health_avg, health_confidence, health_forecast_list = "Error", 0, 0, []

        perf_target = df_ts["PerformanceScore"]
        perf_exog = df_ts[["CuttingSpeed", "FeedRate", "ToolWearRate"]]
        try:
            perf_model = SARIMAX(perf_target, exog=perf_exog, order=(1, 1, 1))
            perf_fit = perf_model.fit(disp=False)
            perf_forecast = perf_fit.forecast(steps=5, exog=perf_exog[-5:])
            perf_avg = round(perf_forecast.mean(), 2)
            perf_trend = "Improving" if perf_avg > perf_target.iloc[-1] else "Declining"
            perf_confidence = round(1 - perf_fit.pvalues.mean(), 2)
            perf_forecast_list = [round(val, 2) for val in perf_forecast.tolist()]
        except Exception as e:
            logger.error(f"Performance forecast error: {e}")
            perf_trend, perf_avg, perf_confidence, perf_forecast_list = "Error", 0, 0, []

        return {
            "health_trend": f"{health_trend} (Avg: {health_avg}/100 over 5 days)",
            "health_forecast": health_forecast_list,
            "performance_trend": f"{perf_trend} (Avg: {perf_avg}/100 over 5 days)",
            "performance_forecast": perf_forecast_list,
            "confidence": round((health_confidence + perf_confidence) / 2, 2)
        }
    
    def process_batch(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not data:
            logger.warning("Empty batch received")
            return []
        processed_records = []
        for record in data:
            try:
                result = self.process_single_record(record)
                processed_records.append(result)
            except Exception as e:
                logger.error(f"Error processing record: {e}")
        return processed_records

    def process_and_save(self, data: List[Dict[str, Any]], output_path: str):
        results = self.process_batch(data)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Saved predictions to {output_path}")
        save_for_training(data, results)

def save_for_training(data: List[Dict[str, Any]], predictions: List[Dict[str, Any]]):
    training_data = [{"input": d, "output": p["prediction"]} for d, p in zip(data, predictions)]
    with open("training_data.json", "a") as f:
        json.dump(training_data, f, indent=4)
        f.write("\n")

def run_continuous_automation():
    global machines, machine_histories, simulated_failure
    processor = BatchProcessor()
    
    while True:
        try:
            for machine_id in machines.keys():
                failure_type = simulated_failure if simulated_failure else None
                record = processor.generator.generate_single_record(failure_type, machine_id)
                result = processor.process_single_record(record)
                machines[machine_id] = result
                history_entry = {
                    "timestamp": result["prediction"]["timestamp"],
                    "failure_type": result["prediction"]["failure_type"],
                    "action": result["prediction"]["actions"][0],
                    "forecast": f"Health: {result['prediction']['forecast']['health_trend']}, Perf: {result['prediction']['forecast']['performance_trend']}"
                }
                machine_histories[machine_id].append(history_entry)
                if len(machine_histories[machine_id]) > 5:
                    machine_histories[machine_id].pop(0)
                logger.info(f"{machine_id} - Prediction: {result['prediction']['failure_type']}, Forecast: Health {result['prediction']['forecast']['health_trend']}, Perf {result['prediction']['forecast']['performance_trend']}")
        except Exception as e:
            logger.error(f"Error in automation: {e}")
        time.sleep(10)

def run_automation(num_records: int = 100, interval_minutes: int = 60):
    generator = RandomDataGenerator()
    processor = BatchProcessor()

    def job():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"predictions/auto_predictions_{timestamp}.json"
        batch_data = generator.generate_batch(num_records)
        processor.process_and_save(batch_data, output_path)

    job()
    schedule.every(interval_minutes).minutes.do(job)
    logger.info(f"Scheduled automation to run every {interval_minutes} minutes")
    while True:
        schedule.run_pending()
        time.sleep(1)

@app.on_event("startup")
async def startup_event():
    threading.Thread(target=run_continuous_automation, daemon=True).start()
    threading.Thread(target=run_automation, args=(100, 60), daemon=True).start()
    logger.info("Scheduled batch processing started in background")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, machine_id: str = "Machine1"):
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "result": machines.get(machine_id),
            "history": machine_histories.get(machine_id, []),
            "machines": machines.keys(),
            "settings": settings
        }
    )

@app.get("/simulate/{failure_type}")
async def simulate_failure(failure_type: str):
    global simulated_failure
    if failure_type == "None":
        simulated_failure = None
    elif failure_type in FAILURE_TYPES:
        simulated_failure = failure_type
    return {"message": f"Simulating {failure_type if simulated_failure else 'random'} next"}

@app.get("/download_history")
async def download_history_report(machine_id: str, from_date: str, to_date: str):
    if machine_id not in machines:
        raise HTTPException(status_code=404, detail="Machine not found")
    try:
        from_dt = datetime.fromisoformat(from_date.replace("Z", "+00:00"))
        to_dt = datetime.fromisoformat(to_date.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")
    
    # Log history for debugging
    logger.info(f"History for {machine_id}: {machine_histories.get(machine_id, [])}")
    
    # Filter in-memory history
    if machine_id not in machine_histories or not machine_histories[machine_id]:
        filtered_history = []
    else:
        filtered_history = [
            entry for entry in machine_histories[machine_id]
            if from_dt <= datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S") <= to_dt
        ]
    
    # Generate PDF report (even if empty)
    file_path = f"reports/history_report_{machine_id}_{from_date.split('T')[0]}_to_{to_date.split('T')[0]}.pdf"
    os.makedirs("reports", exist_ok=True)
    
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(f"History Report for {machine_id}", styles['Title']))
    elements.append(Paragraph(f"From: {from_date}", styles['Normal']))
    elements.append(Paragraph(f"To: {to_date}", styles['Normal']))
    elements.append(Spacer(1, 12))

    if not filtered_history:
        elements.append(Paragraph("No history data found for the selected date range.", styles['Normal']))
    else:
        table_data = [["Timestamp", "Failure Type", "Action", "Forecast"]]
        for entry in filtered_history:
            table_data.append([
                entry["timestamp"],
                entry["failure_type"],
                entry["action"],
                entry["forecast"]
            ])
        table = Table(table_data)
        table.setStyle([
            ('BACKGROUND', (0, 0), (-1, 0), '#203a43'),
            ('TEXTCOLOR', (0, 0), (-1, 0), '#ffffff'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, '#000000'),
            ('BACKGROUND', (0, 1), (-1, -1), '#f0f0f0'),
        ])
        elements.append(table)
    
    doc.build(elements)
    
    return FileResponse(file_path, filename=f"history_report_{machine_id}_{from_date.split('T')[0]}_to_{to_date.split('T')[0]}.pdf", media_type="application/pdf")

@app.post("/settings")
async def update_settings(vibration_threshold: float = Form(...)):
    settings["alert_thresholds"]["VibrationLevel"] = vibration_threshold
    return {"message": "Settings updated"}