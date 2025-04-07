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

# Ingreso de datos por el usuario
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
    
    # Obtener las probabilidades de cada clase
    probabilidades = modelo.predict_proba(data)
    
    # Mostrar las probabilidades de cada clase
    st.write("### Probabilidades de cada clase:")
    for clase, probabilidad in zip(label_encoder.classes_, probabilidades[0]):
        st.write(f"{clase}: {probabilidad:.2f}")

    try:
        # Verificar si la predicción es un array de cadenas (lo cual no es lo esperado)
        if isinstance(prediccion[0], str):  # Si el valor es una cadena (e.g., 'B')
            # Convertir la predicción a su índice usando label_encoder
            prediccion = label_encoder.transform(prediccion)

        # Ahora que la predicción es un índice, usamos inverse_transform para obtener la clase
        clase_predicha = label_encoder.inverse_transform(prediccion)[0]

        # Mostrar la predicción
        st.write("### Resultado de la predicción:")
        st.write(f"La balanza se inclinará hacia: **{clase_predicha}**")
    except ValueError as e:
        st.write(f"Error en la predicción: {str(e)}")
