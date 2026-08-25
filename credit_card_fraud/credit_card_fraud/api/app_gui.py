import random
import numpy as np
import requests
import streamlit as st

st.set_page_config(page_title="Detección de Fraude ML", page_icon="💳", layout="centered")

API_URL = "http://127.0.0.1:8000/predict"

st.title("💳 Monitor de Detección de Fraude")
st.write("Generador dinámico de transacciones para pruebas de estrés de la API.")

# Base estadística para simular transacciones reales con variación aleatoria
BASE_NORMAL = {
    "Time": 85000.0, "V1": -0.42, "V2": 0.15, "V3": 1.10, "V4": -0.20,
    "V5": 0.30, "V6": -0.10, "V7": 0.20, "V8": 0.05, "V9": 0.10, "V10": -0.05,
    "V11": -0.30, "V12": 0.10, "V13": 0.20, "V14": 0.00, "V15": 0.50, "V16": -0.10,
    "V17": 0.05, "V18": 0.00, "V19": -0.10, "V20": 0.02, "V21": -0.01, "V22": 0.05,
    "V23": -0.02, "V24": 0.01, "V25": 0.03, "V26": -0.05, "V27": 0.01, "V28": -0.01,
    "Amount": 120.00
}

BASE_FRAUDE = {
    "Time": 30000.0, "V1": -5.10, "V2": 4.80, "V3": -6.50, "V4": 5.20,
    "V5": -4.10, "V6": -2.80, "V7": -5.90, "V8": 3.10, "V9": -3.80, "V10": -6.20,
    "V11": 4.50, "V12": -5.80, "V13": -0.90, "V14": -7.10, "V15": 0.20, "V16": -3.90,
    "V17": -6.00, "V18": -2.10, "V19": 1.20, "V20": 0.80, "V21": 1.10, "V22": -0.30,
    "V23": -0.50, "V24": 0.10, "V25": 0.40, "V26": 0.30, "V27": 0.80, "V28": -0.20,
    "Amount": 350.00
}

def generar_transaccion_aleatoria(perfil="normal"):
    """Genera datos dinámicos aplicando variaciones aleatorias a las variables."""
    base = BASE_NORMAL if perfil == "normal" else BASE_FRAUDE
    muestra_dinamica = {}
    
    # Variación aleatoria para las componentes V1-V28
    for key, val in base.items():
        if key in ["Time", "Amount"]:
            # El monto varía aleatoriamente dentro de un rango realista
            if perfil == "normal":
                muestra_dinamica["Amount"] = round(float(np.random.uniform(5.0, 850.0)), 2)
            else:
                muestra_dinamica["Amount"] = round(float(np.random.uniform(1.0, 2500.0)), 2)
            muestra_dinamica["Time"] = round(float(np.random.uniform(0.0, 172792.0)), 1)
        else:
            # Añade un pequeño ruido gaussiano (desviación de 15%) a la variable
            ruido = np.random.normal(0, 0.15)
            muestra_dinamica[key] = round(float(val + ruido), 6)
            
    return muestra_dinamica

# --- Interfaz de Streamlit ---
tipo_perfil = st.radio(
    "Selecciona el perfil de simulación:",
    ["Simular Perfil Normal", "Simular Perfil Anómalo / Fraude"],
    horizontal=True
)

if st.button("🎲 Generar Transacción Aleatoria Unica", type="primary"):
    perfil = "normal" if "Normal" in tipo_perfil else "fraude"
    st.session_state["transaccion_activa"] = generar_transaccion_aleatoria(perfil)

if "transaccion_activa" in st.session_state:
    datos_actuales = st.session_state["transaccion_activa"]
    
    st.subheader("📋 Datos Sintéticos Generados Dinámicamente:")
    col_a, col_b = st.columns(2)
    col_a.metric("Amount (Monto)", f"${datos_actuales['Amount']} USD")
    col_b.metric("Time (Segundos)", datos_actuales['Time'])
    
    with st.expander("Ver detalle completo de V1-V28"):
        st.json(datos_actuales)
        
    if st.button("🚀 Enviar a la API para Predicción"):
        try:
            res = requests.post(API_URL, json={"data": [datos_actuales]}, timeout=5)
            if res.status_code == 200:
                result = res.json()["results"][0]
                st.divider()
                if result["is_fraud"]:
                    st.error(f"🚨 DIAGNÓSTICO: {result['diagnosis']}")
                else:
                    st.success(f"✅ DIAGNÓSTICO: {result['diagnosis']}")
                
                m1, m2 = st.columns(2)
                m1.metric("Error de Reconstrucción", result["reconstruction_error"])
                m2.metric("Confianza", f"{result['confidence_score']}%")
            else:
                st.error(f"Error en la API: {res.status_code}")
        except Exception as e:
            st.error(f"Error de conexión con FastAPI: {e}")