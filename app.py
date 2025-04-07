import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Título de la app
st.title("Predicción de Balance Scale")

# Cargar el modelo, el codificador y el scaler
modelo = joblib.load("final_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Campos de entrada para los atributos del Balance Scale
feats = ['Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance']

# Ingreso de datos por el usuario 
left_weight = st.slider("Peso del lado izquierdo", min_value=1, max_value=5, value=3)
left_distance = st.slider("Distancia del lado izquierdo", min_value=1, max_value=5, value=3)
right_weight = st.slider("Peso del lado derecho", min_value=1, max_value=5, value=3)
right_distance = st.slider("Distancia del lado derecho", min_value=1, max_value=5, value=3)

# Crear DataFrame con los valores ingresados
data = pd.DataFrame({
    feats[0]: [left_weight],
    feats[1]: [left_distance],
    feats[2]: [right_weight],
    feats[3]: [right_distance]
})

# Botón para predecir
if st.button("Predecir"):
    # Escalar los datos
    data_scaled = scaler.transform(data)

    # Realizar la predicción
    prediccion = modelo.predict(data_scaled)
    
    # Invertir la transformación del label encoder para obtener el valor original
    clase_predicha = label_encoder.inverse_transform(prediccion)[0]

    # Mostrar la predicción
    st.write("### Predicción del Modelo")
    st.write(f"La balanza se inclinará hacia: **{clase_predicha}**")
