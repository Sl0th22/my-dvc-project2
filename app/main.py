from fastapi import FastAPI
import mlflow.pyfunc
import numpy as np
from pydantic import BaseModel

app = FastAPI()

model = mlflow.pyfunc.load_model("./model_artifact")


class PredictRequest(BaseModel):
  features: list[list[float]]


@app.post("/predict")
def predict(payload: PredictRequest):
  data = np.array(payload.features)
  predictions = model.predict(data)
  return {"predictions": predictions.tolist()}