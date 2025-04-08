import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Título de la app
st.title("Predicción de Balance Scale")

# Cargar el modelo, el codificador y el scaler
modelo = joblib.load("final_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load('scaler.pkl')

# Campos de entrada para los atributos del Balance Scale
feats = ['Left Weight', 'Left Distance', 'Right Weight', 'Right Distance']

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

# Escalar los datos
data_scaled = scaler.transform(data)

# Botón para predecir
if st.button("Predecir"):
    # Obtener las probabilidades de cada clase
    probabilidades = modelo.predict_proba(data_scaled)
    
    # Predecir la clase con la mayor probabilidad
    prediccion = np.argmax(probabilidades, axis=1)
    
    # Obtener la clase predicha
    clase_predicha = label_encoder.inverse_transform(prediccion)[0]

    # Mostrar las probabilidades
    st.write(f"**Probabilidades**: {dict(zip(label_encoder.classes_, probabilidades[0]))}")

    # Mostrar la predicción
    if clase_predicha == 'L':
        st.write("### Predicción: La balanza se inclina hacia la izquierda")
    elif clase_predicha == 'R':
        st.write("### Predicción: La balanza se inclina hacia la derecha")
    else:
        st.write("### Predicción: La balanza está balanceada")
