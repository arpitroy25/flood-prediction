import streamlit as st
import pandas as pd
import pickle
from weather import get_weather

st.title("🌊 Automatic Flood Prediction Dashboard")


with open("model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
scaler = model_data.get("scaler")
needs_scaling = model_data.get("needs_scaling", False)

city = st.text_input("Enter City Name")

if st.button("Predict Flood Risk"):
    if not city.strip():
        st.warning("Please enter a city name.")
        st.stop()
    
    city = city.strip().title()

    weather = get_weather(city)

    if weather is None:
        st.error("Unable to fetch weather data. Check city name or API key.")
    else:
        rainfall = weather["rainfall"]
        humidity = weather["humidity"]
        temperature = weather["temperature"]

        water_level = rainfall * 0.03

        st.subheader("🌦 Live Weather Data")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("🌧 Rainfall", f"{rainfall} mm")
        col2.metric("💧 Humidity", f"{humidity}%")
        col3.metric("🌡 Temperature", f"{temperature:.1f}°C")
        col4.metric("🌊 Water Level", f"{water_level:.2f} m")

        data = pd.DataFrame(
            [[rainfall, water_level, humidity, temperature]],
            columns=["rainfall", "water_level", "humidity", "temperature"]
        )

        if needs_scaling and scaler:
            data = scaler.transform(data)

        result = model.predict(data)

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(data)[0][1]
        else:
            probability = 0.5

        if probability < 0.25:
            risk = "🟢 Low Risk"
        elif probability < 0.50:
            risk = "🟡 Medium Risk"
        elif probability < 0.75:
            risk = "🔴 High Risk"
        else:
            risk = "🟣 Critical Risk"

        st.subheader(f"Risk Level: {risk}")
        st.progress(probability)

        st.metric(
            "Flood Probability",
            f"{probability * 100:.1f}%"
        )

        st.subheader("💡 Flood Risk Analysis")

        if result[0] == 1:
            st.error("⚠️ Flood Expected")
        else:
            st.success("✅ No Flood Expected")

        st.info(f"Prediction Confidence: {max(probability, 1-probability)*100:.1f}%")

        