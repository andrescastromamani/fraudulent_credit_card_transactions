# Integración con MLflow

## 1. Objetivo

Se conectó el pipeline de detección de fraude con MLflow para guardar y consultar:

- Los experimentos de entrenamiento.
- Los hiperparámetros utilizados.
- Las métricas de entrenamiento y validación.
- Las métricas finales sobre el conjunto de prueba.
- Los modelos Keras generados.
- Los checkpoints y los archivos de evaluación.

La integración se basó en el ejemplo de `AdvancedMLFlow.ipynb`, pero se adaptó al proyecto real. El ejemplo utiliza un `RandomForestClassifier` y un wrapper `mlflow.pyfunc`; este proyecto utiliza dos modelos de TensorFlow/Keras, por lo que se conservó el autologging nativo de TensorFlow.

## 2. Backend local de MLflow

Inicialmente se intentó utilizar la carpeta `mlruns` como backend de tracking. La versión instalada de MLflow rechazó este formato porque el filesystem backend se encuentra en modo de mantenimiento.

Por ese motivo se configuró una base de datos SQLite:

```text
sqlite:///D:/PROYECTOMODULO5/house_price_project/credit_card_fraud/mlflow.db
```

La configuración central se encuentra en `credit_card_fraud/config.py`:

```python
MLFLOW_DB_PATH = PACKAGE_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
MLFLOW_EXPERIMENT = "fraud_detection"
```

De esta forma, los metadatos de las corridas se almacenan en `mlflow.db`, mientras que los artefactos quedan asociados a las corridas de MLflow.

## 3. Corrección de rutas del proyecto

También se separaron correctamente las rutas del paquete y de la raíz del proyecto:

- `PACKAGE_ROOT`: carpeta `credit_card_fraud`.
- `PROJ_ROOT`: carpeta `house_price_project`.
- Datos: `house_price_project/data`.
- Modelos e informes: `credit_card_fraud/models` y `credit_card_fraud/reports`.
- Base de datos de MLflow: `credit_card_fraud/mlflow.db`.

Esto permite que el pipeline encuentre el dataset y registre los archivos generados en las ubicaciones esperadas.

## 4. Tracking del modelo MLP

En `credit_card_fraud/modeling/train.py`, el método `MLPModel.train()` realiza estas acciones:

1. Configura el URI de tracking de MLflow.
2. Selecciona el experimento `fraud_detection`.
3. Abre una corrida llamada `MLP_Supervisado_Training`.
4. Activa `mlflow.tensorflow.autolog(log_models=True)`.
5. Registra los parámetros propios del modelo:
   - `learning_rate`.
   - `dropout_rate`.
   - `input_dim`.
   - `epochs`.
   - `batch_size`.
   - `validation_split`.
6. Ejecuta el entrenamiento de Keras.
7. Deja que MLflow registre automáticamente las métricas por época y el modelo.

El MLP es el clasificador supervisado que produce una probabilidad de fraude.

## 5. Tracking del autoencoder

El método `AutoencoderModel.train()` utiliza el mismo experimento y el mismo backend, pero crea una corrida llamada `Autoencoder_Training`.

Se registran los siguientes parámetros:

- `encoding_dim`.
- `input_dim`.
- `epochs`.
- `batch_size`.
- `validation_split`.

También se activa el autologging de TensorFlow para guardar las métricas de pérdida, las métricas de validación y el modelo Keras generado.

El autoencoder se entrena únicamente con transacciones normales. Después, el error de reconstrucción se utiliza como puntuación de anomalía.

## 6. Corrida de evaluación comparativa

El archivo `run_pipeline.py` abre una tercera corrida llamada `Evaluation_Comparativa`, después de terminar ambos entrenamientos.

Esta corrida registra los parámetros generales del experimento:

- `epochs`.
- `batch_size`.
- `seed`.
- `test_rows`.

Para cada modelo se registran las siguientes métricas:

- Threshold óptimo.
- Accuracy.
- Precision.
- Recall.
- F1.
- ROC-AUC.
- PR-AUC.

Los nombres de las métricas incluyen el modelo para poder comparar ambos resultados dentro de la misma corrida. Por ejemplo:

```text
_dl_1_mlp_supervisado_pr_auc
_dl_2_autoencoder_pr_auc
```

El threshold se calcula con el método `optimal_threshold()`, que busca maximizar el F1 sobre las predicciones del conjunto de prueba.

## 7. Artefactos registrados

La corrida `Evaluation_Comparativa` guarda estos archivos:

```text
evaluation/model_results.csv
evaluation/pr_curves.png
checkpoints/model_mlp.keras
checkpoints/model_autoencoder.keras
```

`model_results.csv` contiene la tabla comparativa final y `pr_curves.png` contiene las curvas Precision-Recall de ambos modelos.

Los modelos Keras también se guardan localmente mediante los checkpoints configurados:

```text
credit_card_fraud/models/model_mlp.keras
credit_card_fraud/models/model_autoencoder.keras
```

## 8. Ejecución del pipeline

Desde la carpeta `credit_card_fraud` se ejecuta:

```powershell
python run_pipeline.py
```

El pipeline realiza este flujo:

```text
Carga de datos
    -> Ingeniería de características
    -> Entrenamiento MLP + corrida MLflow
    -> Entrenamiento Autoencoder + corrida MLflow
    -> Predicciones sobre test
    -> Evaluación comparativa + corrida MLflow
    -> Registro de métricas y artefactos
```

## 9. Abrir la interfaz de MLflow

Para iniciar la interfaz web usando SQLite:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Si `mlflow` no está disponible directamente en el PATH, se puede utilizar el entorno virtual:

```powershell
& D:\PROYECTOMODULO5\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Después se abre en el navegador:

```text
http://127.0.0.1:5000
```

En la interfaz se debe seleccionar el experimento `fraud_detection`. Allí se pueden revisar las corridas de entrenamiento y la corrida `Evaluation_Comparativa`.

## 10. Verificación realizada

Se realizaron estas comprobaciones:

1. Compilación sintáctica de los módulos modificados mediante `compileall`.
2. Creación correcta del experimento `fraud_detection` en SQLite.
3. Ejecución completa de `run_pipeline.py`.
4. Confirmación de corridas finalizadas correctamente para MLP, autoencoder y evaluación.
5. Confirmación de los artefactos:

```text
evaluation/model_results.csv
evaluation/pr_curves.png
```

La ejecución produjo, entre otros resultados, estos valores de evaluación:

| Modelo | ROC-AUC | PR-AUC | F1-Score |
|---|---:|---:|---:|
| MLP supervisado | 0.982863 | 0.686782 | 0.738739 |
| Autoencoder | 0.963274 | 0.276137 | 0.426332 |

## 11. Relación con el notebook de ejemplo

El notebook presenta conceptos avanzados como:

- Tracking remoto y backend store.
- Registro de parámetros y artefactos.
- Evaluación automatizada.
- Model Registry.
- Modelos personalizados con `mlflow.pyfunc`.

En esta primera integración se implementaron directamente los elementos necesarios para el pipeline actual:

- Backend SQLite local.
- Experimento centralizado.
- Autologging de TensorFlow/Keras.
- Logging manual de parámetros adicionales.
- Logging manual de métricas finales.
- Logging de modelos, checkpoints y resultados.

El wrapper `EnterpriseDecisionWrapper` del notebook no se incorporó porque espera un modelo con `predict_proba()`, mientras que los modelos utilizados aquí son redes Keras con `predict()`. Para añadir un modelo servido con `mlflow.pyfunc` habría que crear un wrapper específico para las salidas del MLP y del autoencoder.
