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

# Definir las columnas que el modelo espera
columnas_esperadas = ['Left Weight', 'Left Distance', 'Right Weight', 'Right Distance']

# Ingreso de datos por el usuario
left_weight = st.slider("Peso del lado izquierdo", min_value=1, max_value=5, value=3)
left_distance = st.slider("Distancia del lado izquierdo", min_value=1, max_value=5, value=3)
right_weight = st.slider("Peso del lado derecho", min_value=1, max_value=5, value=3)
right_distance = st.slider("Distancia del lado derecho", min_value=1, max_value=5, value=3)

# Crear DataFrame con los valores ingresados, asegurando que las columnas coincidan
data = pd.DataFrame({
    'Left Weight': [left_weight],
    'Left Distance': [left_distance],
    'Right Weight': [right_weight],
    'Right Distance': [right_distance]
})

# Mostrar los nombres de las columnas del DataFrame
st.write("Columnas del DataFrame ingresado:", data.columns)

# Asegurarse de que las columnas estén en el orden correcto
# Eliminar espacios al principio y al final de los nombres de las columnas
data.columns = data.columns.str.strip()

# Verificar si las columnas coinciden con las esperadas
if list(data.columns) != columnas_esperadas:
    st.error("¡Error! Las columnas del DataFrame no coinciden con las esperadas.")
    st.write("Columnas esperadas:", columnas_esperadas)
    st.write("Columnas del DataFrame:", list(data.columns))
else:
    # Si las columnas son correctas, proceder con la predicción
    st.write("Datos ingresados:", data)

    # Escalar los datos usando el scaler entrenado
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
