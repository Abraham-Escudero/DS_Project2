import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Título de la app
st.title("Predicción de Balance Scale")

# Cargar el modelo, el codificador de etiquetas y el escalador
modelo = joblib.load("final_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Ingreso de datos por el usuario (ajusta estos nombres si tus columnas son distintas)
left_weight = st.slider("Peso del lado izquierdo", min_value=1, max_value=5, value=3)
left_distance = st.slider("Distancia del lado izquierdo", min_value=1, max_value=5, value=3)
right_weight = st.slider("Peso del lado derecho", min_value=1, max_value=5, value=3)
right_distance = st.slider("Distancia del lado derecho", min_value=1, max_value=5, value=3)

# Crear DataFrame con los datos de entrada
data = pd.DataFrame({
    'Left-Weight': [left_weight],
    'Left-Distance': [left_distance],
    'Right-Weight': [right_weight],
    'Right-Distance': [right_distance]
})

# Escalar los datos
data_scaled = scaler.transform(data)

# Botón para predecir
if st.button("Predecir"):
    # Realizar la predicción
    prediccion = modelo.predict(data_scaled)

    # Decodificar la predicción a su etiqueta original
    clase_predicha = label_encoder.inverse_transform(prediccion)[0]

    # Mostrar el resultado
    st.write("### Resultado de la predicción:")
    st.write(f"La balanza se inclinará hacia: **{clase_predicha}**")

    # Si deseas obtener la probabilidad de la predicción (solo si el modelo tiene `probability=True`)
    probabilidades = modelo.predict_proba(data_scaled)
    st.write("### Probabilidades de la predicción:")
    st.write(f"Probabilidad de izquierda: {probabilidades[0][0]:.4f}")
    st.write(f"Probabilidad de derecha: {probabilidades[0][1]:.4f}")
    st.write(f"Probabilidad de estar balanceada: {probabilidades[0][2]:.4f}")
