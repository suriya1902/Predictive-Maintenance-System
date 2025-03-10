from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the saved model and recommendation system
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the recommendation system pickle file
with open('recommendation_system.pkl', 'rb') as f:
    recommendation_system = pickle.load(f)
    df = recommendation_system['df']
    scaler = recommendation_system['scaler']
    X_scaled = recommendation_system['X_scaled']

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

FAILURE_TYPES = ['General Failure','Excessive Tool Wear','Overheating','Vibration Issue']
ACTIONS = ['Examine machine alignment and spindle condition',
       'Inspect cooling system and check lubrication', 'No action required',
       'Replace tool and adjust feed rate',
       'Schedule maintenance check immediately']

# Image paths mapped to each failure type
IMAGE_PATHS = {
    "Excessive Tool Wear": "/static/images/wear and tear.png",
    "General Failure": "/static/images/Predictive-maintenance.png",
    "Vibration Issue": "/static/images/vibration.png",
    "Overheating": "/static/images/overheating.png"
}

class PredictionInput(BaseModel):
    output: List[int]

def map_prediction(output: List[int]):
    """
    Maps the model's output to a single failure type and corresponding actions.
    """
    failures = [FAILURE_TYPES[i] for i in range(len(FAILURE_TYPES)) if output[i] == 1]

    # Select only the first failure type if multiple are predicted
    predicted_failure = failures[0] if failures else "No Failures Detected"

    # Select corresponding action(s) for the predicted failure
    actions = [ACTIONS[i] for i in range(len(ACTIONS)) if output[i + len(FAILURE_TYPES)] == 1]

    recommendation_text = "Actions to be Taken: " + ", ".join(actions) if actions else "No Actions Required"

    return predicted_failure, recommendation_text


# Functions for converting string to integer values for prediction
def machinealignmentstatus(value):
    return 1 if value == "Aligned" else 0

def operationmode(value):
    auto, semi, man = 0, 0, 0
    if value == "Auto":
        auto = 1
    elif value == "Semi-Automatic":
        semi = 1
    elif value == "Manual":
        man = 1
    return auto, semi, man

def spindlecondition(value):
    return 1 if value == "Good" else 0

def tooltype(value):
    ca, ce, hss = 0, 0, 0
    if value == "HSS":
        hss = 1
    elif value == "Carbide":
        ca = 1
    elif value == "Ceramic":
        ce = 1
    return ca, ce, hss

def lubricationcondition(value):
    fair, good, poor = 0, 0, 0
    if value == "Fair":
        fair = 1
    elif value == "Good":
        good = 1
    elif value == "Poor":
        poor = 1
    return fair, good, poor

def materialtype(value):
    alu, ste, tit = 0, 0, 0
    if value == "Titanium":
        tit = 1
    elif value == "Aluminium":
        alu = 1
    elif value == "Steel":
        ste = 1
    return alu, ste, tit

# Function to handle predictions
def predict_failure(OperationTime: float, CuttingSpeed: float, FeedRate: float, ToolDiameter: float, SpindleSpeed: float, MotorCurrent: float,
                   PowerConsumption: float, LubricationLevel: float, CoolingSystemEfficiency: float, CycleTime: float, MachineAge: float,
                   PartDefectRate: float, MaintenanceFrequency: float, VibrationLevel: float, MachineHealthStatus: float, ToolWearRate: float,
                   CuttingTemperature: float, MaterialHardness: float, PressureLevel: float, MachineStability: float, MachineAlignmentStatus: str,
                   PowerSurgeRate: float, MachineCalibration: float, ToolChangeInterval: float, SpindleHealthStatus: float, LubricationCondition: str,
                   CoolantFlowRate: float, CuttingForce: float, OperationMode: str, MachineLoad: float, SpindleTorque: float, SpindleCondition: str,
                   PowerSupplyVoltage: float, ToolType: str, MaterialType: str):
   
    # Convert categorical variables
    MachineAlignmentStatus_int = machinealignmentstatus(MachineAlignmentStatus)
    SpindleCondition_int = spindlecondition(SpindleCondition)
    auto, semi, man = operationmode(OperationMode)
    ca, ce, hss = tooltype(ToolType)
    alu, ste, tit = materialtype(MaterialType)
    fair, good, poor = lubricationcondition(LubricationCondition)

    # Create feature vector with 38 features
    features = np.array([[ 
        OperationTime, CuttingSpeed, FeedRate, ToolDiameter, SpindleSpeed, MotorCurrent, PowerConsumption, LubricationLevel, CoolingSystemEfficiency, CycleTime, MachineAge, PartDefectRate, MaintenanceFrequency, 
        VibrationLevel, MachineHealthStatus, ToolWearRate, CuttingTemperature, MaterialHardness, PressureLevel, MachineStability, PowerSurgeRate, MachineCalibration, ToolChangeInterval, SpindleHealthStatus, 
        CoolantFlowRate, CuttingForce, MachineLoad, SpindleTorque, PowerSupplyVoltage, ca, ce, hss, alu, ste, tit, fair, good, poor  ]])

    # Predict using the model
    prediction = model.predict(features)
    return prediction

# Define the target columns for failure types
target_columns = ['Excessive Tool Wear','General Failure','Overheating','Vibration Issue']

# Add a new route for the root URL (GET request)
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction route - handles form submission
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    OperationTime: float = Form(...), CuttingSpeed: float = Form(...), FeedRate: float = Form(...), ToolDiameter: float = Form(...), SpindleSpeed: float = Form(...), MotorCurrent: float = Form(...),
    PowerConsumption: float = Form(...), LubricationLevel: float = Form(...), CoolingSystemEfficiency: float = Form(...), CycleTime: float = Form(...), MachineAge: float = Form(...), 
    PartDefectRate: float = Form(...), MaintenanceFrequency: float = Form(...), VibrationLevel: float = Form(...), MachineHealthStatus: float = Form(...), ToolWearRate: float = Form(...), 
    CuttingTemperature: float = Form(...), MaterialHardness: float = Form(...), PressureLevel: float = Form(...), MachineStability: float = Form(...), MachineAlignmentStatus: str = Form(...),
    PowerSurgeRate: float = Form(...), MachineCalibration: float = Form(...), ToolChangeInterval: float = Form(...), SpindleHealthStatus: float = Form(...), LubricationCondition: str = Form(...), 
    CoolantFlowRate: float = Form(...), CuttingForce: float = Form(...), OperationMode: str = Form(...), MachineLoad: float = Form(...), SpindleTorque: float = Form(...), SpindleCondition: str = Form(...), 
    PowerSupplyVoltage: float = Form(...), ToolType: str = Form(...), MaterialType: str = Form(...), request: Request = None):

    # Predict using the loaded model
    prediction = predict_failure(
        OperationTime, CuttingSpeed, FeedRate, ToolDiameter, SpindleSpeed, MotorCurrent, PowerConsumption, LubricationLevel, CoolingSystemEfficiency, CycleTime, MachineAge, PartDefectRate, MaintenanceFrequency, 
        VibrationLevel, MachineHealthStatus, ToolWearRate, CuttingTemperature, MaterialHardness, PressureLevel, MachineStability, MachineAlignmentStatus, PowerSurgeRate, MachineCalibration, ToolChangeInterval, 
        SpindleHealthStatus, LubricationCondition, CoolantFlowRate, CuttingForce, OperationMode, MachineLoad, SpindleTorque, SpindleCondition, PowerSupplyVoltage, ToolType, MaterialType)

    # Map the prediction to failure type and actions
    predicted_failure, recommendation_text = map_prediction(prediction[0])

    # Get recommendations based on the predicted failure type
    recommendations = []
    if predicted_failure != "No Failures Detected":
        machine_indices = df[df[predicted_failure] == 1].index.tolist()

        if machine_indices:
            machine_index = machine_indices[0]
            cos_sim = cosine_similarity(X_scaled)
            similar_machines = np.argsort(cos_sim[machine_index])[-4:-1]
            recommended_failures = df.iloc[similar_machines][target_columns].sum().sort_values(ascending=False)
            recommended_failures = recommended_failures.drop(predicted_failure, errors='ignore')
            recommendations = [
                {"name": failure, "image": IMAGE_PATHS.get(failure, "/static/images/default.png")}
                for failure in recommended_failures.index.tolist()[:2]
            ]
   
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "prediction": f"Failure Type: {predicted_failure} - {recommendation_text}",
            "recommendations": recommendations,
        }
    )