import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tensorflow import keras

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from credit_card_fraud.config import AUTOENCODER_MODEL_PATH, MLP_MODEL_PATH

# ==========================================
# 1. INICIALIZACION DE FASTAPI Y CARGA
# ==========================================
app = FastAPI(
    title="API de Deteccion de Fraude con Tarjetas de Credito",
    description="API MLOps para inferencia de modelos de deep learning (MLP Supervisado + Autoencoder).",
    version="1.0.0",
)

mlp_model: keras.Model | None = None
autoencoder_model: keras.Model | None = None
MODEL_INFO = {
    "mlp": str(MLP_MODEL_PATH),
    "autoencoder": str(AUTOENCODER_MODEL_PATH),
}


@app.on_event("startup")
def startup_event():
    global mlp_model, autoencoder_model
    try:
        mlp_model = keras.models.load_model(MLP_MODEL_PATH)
        print(f"[FastAPI] MLP cargado desde {MLP_MODEL_PATH}")
    except Exception as e:
        print(f"[FastAPI ERROR] No se pudo cargar MLP: {e}")
    try:
        autoencoder_model = keras.models.load_model(AUTOENCODER_MODEL_PATH)
        print(f"[FastAPI] Autoencoder cargado desde {AUTOENCODER_MODEL_PATH}")
    except Exception as e:
        print(f"[FastAPI ERROR] No se pudo cargar Autoencoder: {e}")


# ==========================================
# 2. ESQUEMAS DE ENTRADA (Pydantic)
# ==========================================
class TransactionFeatures(BaseModel):
    time: float = Field(..., alias="Time")
    v1: float = Field(..., alias="V1")
    v2: float = Field(..., alias="V2")
    v3: float = Field(..., alias="V3")
    v4: float = Field(..., alias="V4")
    v5: float = Field(..., alias="V5")
    v6: float = Field(..., alias="V6")
    v7: float = Field(..., alias="V7")
    v8: float = Field(..., alias="V8")
    v9: float = Field(..., alias="V9")
    v10: float = Field(..., alias="V10")
    v11: float = Field(..., alias="V11")
    v12: float = Field(..., alias="V12")
    v13: float = Field(..., alias="V13")
    v14: float = Field(..., alias="V14")
    v15: float = Field(..., alias="V15")
    v16: float = Field(..., alias="V16")
    v17: float = Field(..., alias="V17")
    v18: float = Field(..., alias="V18")
    v19: float = Field(..., alias="V19")
    v20: float = Field(..., alias="V20")
    v21: float = Field(..., alias="V21")
    v22: float = Field(..., alias="V22")
    v23: float = Field(..., alias="V23")
    v24: float = Field(..., alias="V24")
    v25: float = Field(..., alias="V25")
    v26: float = Field(..., alias="V26")
    v27: float = Field(..., alias="V27")
    v28: float = Field(..., alias="V28")
    amount: float = Field(..., alias="Amount")

    class Config:
        allow_population_by_field_name = True


class PredictionRequest(BaseModel):
    data: list[TransactionFeatures]


# ==========================================
# 3. ENDPOINTS
# ==========================================
@app.get("/")
def read_root():
    return {
        "status": "Online",
        "models": {
            "mlp_loaded": mlp_model is not None,
            "autoencoder_loaded": autoencoder_model is not None,
        },
        "model_paths": MODEL_INFO,
    }


@app.post("/predict")
def predict(payload: PredictionRequest):
    if mlp_model is None and autoencoder_model is None:
        raise HTTPException(
            status_code=500,
            detail="Ningun modelo esta cargado en memoria.",
        )

    try:
        input_data = pd.DataFrame([item.dict(by_alias=True) for item in payload.data])

        results = []

        for i in range(len(input_data)):
            row = input_data.iloc[i : i + 1].to_numpy()
            result: dict = {"index": i}

            if mlp_model is not None:
                mlp_prob = float(mlp_model.predict(row, verbose=0).ravel()[0])
                result["mlp_fraud_probability"] = round(mlp_prob, 4)
                result["mlp_prediction"] = "Fraude" if mlp_prob >= 0.5 else "Legitimo"

            if autoencoder_model is not None:
                reconstructed = autoencoder_model.predict(row, verbose=0)
                mse = float(np.mean(np.power(row - reconstructed, 2)))
                result["autoencoder_reconstruction_error"] = round(mse, 6)

            results.append(result)

        return {
            "total_predictions": len(results),
            "results": results,
            "message": "Inferencia completada con exito.",
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error durante la inferencia: {str(e)}")


@app.get("/health")
def health_check():
    return {
        "status": "healthy" if (mlp_model is not None or autoencoder_model is not None) else "degraded",
        "mlp_loaded": mlp_model is not None,
        "autoencoder_loaded": autoencoder_model is not None,
    }
