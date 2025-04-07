import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Título de la app
st.title("Predicción de Balance Scale")

# Cargar el modelo y el codificador
modelo = joblib.load("final_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Mostrar las clases disponibles en el label_encoder
st.write(f"Clases disponibles en el label_encoder: {label_encoder.classes_}")

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

# Botón para predecir
if st.button("Predecir"):
    # Predecir usando el modelo cargado
    prediccion = modelo.predict(data)
    
    # Mostrar la predicción antes de intentar usar el label_encoder
    st.write(f"Predicción del modelo (antes de label_encoder): {prediccion}")

    try:
        # Asegurarnos de que la predicción sea un array de enteros
        prediccion = np.array(prediccion).reshape(-1, 1)  # Convertir en array adecuado si es necesario

        # Invertir la transformación de las predicciones usando el label_encoder
        clase_predicha = label_encoder.inverse_transform(prediccion.flatten())[0]

        # Mostrar la predicción
        st.write("### Resultado de la predicción:")
        st.write(f"La balanza se inclinará hacia: **{clase_predicha}**")
    except ValueError as e:
        st.write(f"Error en la predicción: {str(e)}")
