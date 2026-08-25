# Implementacion de API con FastAPI - Deteccion de Fraude

## Resumen

Se implemento una API REST con FastAPI para servir los modelos de deep learning (MLP Supervisado y Autoencoder) del proyecto de deteccion de fraude con tarjetas de credito.

---

## Pasos Realizados

### 1. Analisis del Proyecto Existente

**Archivos revisados:**
- `credit_card_fraud/config.py` - Configuracion de rutas y MLflow
- `credit_card_fraud/modeling/train.py` - Clases MLPModel y AutoencoderModel
- `credit_card_fraud/modeling/predict.py` - Logica de inferencia y evaluacion
- `credit_card_fraud/dataset.py` - Carga del dataset
- `credit_card_fraud/features.py` - Ingenieria de features
- `data/raw/creditcard.csv` - Dataset con 31 columnas (Time, V1-V28, Amount, Class)

**Modelos encontrados:**
- `models/model_mlp.keras` - MLP Supervisado para clasificacion binaria
- `models/model_autoencoder.keras` - Autoencoder para deteccion de anomalias

### 2. Problemas Identificados en el Archivo Original

El archivo `api/app.py` original tenia varios problemas:

| Problema | Descripcion |
|----------|-------------|
| Titulo incorrecto | Decia "API de Deteccion de Cancer de Mama" |
| Import inexistente | Importaba `src.api.model_loader` que no existe |
| Esquemas incorrectos | Usaba features de cancer (30 features), no de fraude |
| Modelos incorrectos | Solo cargaba un modelo, no los dos del proyecto |
| Logica incorrecta | Usaba `predict_proba()` que no aplica a redes neuronales |

### 3. Solucion Implementada

#### 3.1. Reescritura completa de `api/app.py`

**Cambios realizados:**
- Titulo: "API de Deteccion de Fraude con Tarjetas de Credito"
- Carga ambos modelos (MLP + Autoencoder) al iniciar
- Esquema `TransactionFeatures` con las 30 features correctas (Time, V1-V28, Amount)
- Endpoint `/predict` que retorna:
  - Probabilidad de fraude del MLP
  - Error de reconstruccion del Autoencoder
- Endpoint `/health` para verificar estado de modelos
- Endpoint `/` con informacion del sistema

#### 3.2. Creacion de `api/requirements.txt`

Dependencias necesarias:
```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
tensorflow>=2.15.0
numpy>=1.24.0
pandas>=2.0.0
pydantic>=2.0.0
```

### 4. Estructura Final de la API

```
api/
  app.py           - Aplicacion FastAPI principal
  requirements.txt - Dependencias de la API
```

### 5. Endpoints Disponibles

| Metodo | Ruta | Descripcion |
|--------|------|-------------|
| GET | `/` | Info del sistema y modelos cargados |
| POST | `/predict` | Inferencia con uno o ambos modelos |
| GET | `/health` | Health check de la aplicacion |

### 6. Ejemplo de Uso

#### Iniciar la API

```bash
cd house_price_project/credit_card_fraud
pip install -r api/requirements.txt
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

#### Documentacion interactiva

Abrir en navegador: `http://localhost:8000/docs`

#### Ejemplo de peticion POST

```json
{
  "data": [
    {
      "Time": 0,
      "V1": -1.3598,
      "V2": -0.0728,
      "V3": 2.5363,
      "V4": 1.3782,
      "V5": -0.3383,
      "V6": 0.4624,
      "V7": 0.2396,
      "V8": 0.0987,
      "V9": 0.3638,
      "V10": 0.0908,
      "V11": -0.5516,
      "V12": -0.6178,
      "V13": -0.9914,
      "V14": -0.3112,
      "V15": 1.4682,
      "V16": -0.4704,
      "V17": 0.2080,
      "V18": 0.0258,
      "V19": 0.4040,
      "V20": 0.2514,
      "V21": -0.0183,
      "V22": 0.2778,
      "V23": -0.1105,
      "V24": 0.0669,
      "V25": 0.1285,
      "V26": -0.1891,
      "V27": 0.1336,
      "V28": -0.0211,
      "Amount": 149.62
    }
  ]
}
```

### 7. Arquitectura de la API

```
Request (JSON)
     |
     v
FastAPI Endpoint (/predict)
     |
     v
+------------------+
| Carga de Datos   | -> DataFrame con features
+------------------+
     |
     v
+------------------+     +------------------+
| MLP Model        |     | Autoencoder      |
| (clasificacion)  |     | (anomalias)      |
+------------------+     +------------------+
     |                         |
     v                         v
Probabilidad Fraude    Error Reconstruccion
     |                         |
     v                         v
+------------------+
| Respuesta JSON   |
+------------------+
```

---

## Archivos Modificados

| Archivo | Accion |
|---------|--------|
| `api/app.py` | Reescrito completamente |
| `api/requirements.txt` | Creado nuevo |

---

## Notas Tecnicas

1. **Escalado de datos**: La API recibe datos crudos (Time, Amount sin escalar). Para produccion, se recomienda agregar el escalado `RobustScaler` como en `features.py`.

2. **Threshold del MLP**: Se usa 0.5 como valor por defecto. Se puede ajustar usando `ModelEvaluator.optimal_threshold()` del pipeline de entrenamiento.

3. **Autoencoder**: El endpoint retorna el error de reconstruccion (MSE). Se debe definir un threshold para clasificar como fraude.

4. **Compatibilidad**: La API usa `sys.path` para importar el paquete `credit_card_fraud` correctamente.
