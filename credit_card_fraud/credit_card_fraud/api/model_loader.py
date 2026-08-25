from pathlib import Path
import mlflow
import tensorflow as tf

MODEL_NAME = "Fraud_Detection_Model"

# Obtener la raíz del proyecto (3 niveles arriba desde api/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DB_PATH = BASE_DIR / "mlflow.db"

# Usar URI absoluto para SQLite
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")

def load_model():
    """Carga el modelo más reciente registrado en MLflow omitiendo métricas personalizadas si existen."""
    try:
        print(f"[MLflow Debug] Conectando a BD en: {DB_PATH}")
        experiment = mlflow.get_experiment_by_name("fraud_detection")
        
        if not experiment:
            print("[MLflow Error] No se encontró el experimento 'fraud_detection'.")
            return None
            
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
        if runs.empty:
            print("[MLflow Error] No existen ejecuciones (runs) grabadas en la BD.")
            return None
            
        latest_run_id = runs.iloc[0]["run_id"]
        model_uri = f"runs:/{latest_run_id}/model"
        print(f"[MLflow Debug] Intentando cargar run_id: {latest_run_id}")
        
        # 1. Intentar carga nativa de pyfunc (la forma recomendada por MLflow)
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            print("--> Modelo cargado exitosamente usando mlflow.pyfunc")
            return model
        except Exception as e_pyfunc:
            print(f"[MLflow Warning] pyfunc falló, intentando Keras directo: {e_pyfunc}")

        # 2. Si falla pyfunc, cargar vía Keras desactivando la compilación de métricas estrictas
        artifact_path = mlflow.artifacts.download_artifacts(run_id=latest_run_id, artifact_path="model/data/model.keras")
        model = tf.keras.models.load_model(artifact_path, compile=False)
        print("--> Modelo Keras cargado exitosamente (compile=False)")
        return model

    except Exception as e:
        print(f"[MLflow Error Fatal] Falló al cargar el modelo: {e}")
        return None

def get_model_metadata():
    """Obtiene metadatos sobre la versión y ejecución del modelo."""
    try:
        experiment = mlflow.get_experiment_by_name("fraud_detection")
        if not experiment:
            return {"version": "Desconocida", "run_id": "Desconocido"}
            
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
        if runs.empty:
            return {"version": "Desconocida", "run_id": "Desconocido"}
            
        latest_run = runs.iloc[0]
        return {
            "version": str(latest_run.get("tags.mlflow.runName", "v1.0.0")),
            "run_id": str(latest_run["run_id"])
        }
    except Exception:
        return {"version": "Desconocida", "run_id": "Desconocido"}