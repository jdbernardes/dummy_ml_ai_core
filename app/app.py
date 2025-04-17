from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("model.joblib")

class InputData(BaseModel):
    data: list[list[float]]

@app.post("/predict")
def predict(input: InputData):
    prediction = model.predict(input.data)
    return {"prediction": prediction.tolist()}
