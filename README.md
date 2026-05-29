# flood-prediction
🌊 AI-powered Flood Prediction System using Machine Learning and real-time weather data from OpenWeatherMap API. Predicts flood risk, probability, confidence score, and risk levels through an interactive Streamlit dashboard.


# 🌊 Flood Prediction System using Machine Learning and Live Weather Data

## 📌 Overview

The Flood Prediction System is an AI-powered web application that predicts the possibility of floods using Machine Learning and real-time weather data.

The system fetches live weather information from the OpenWeatherMap API, including rainfall, humidity, and temperature, and uses a trained Machine Learning model to predict flood risk. The application provides flood probability, confidence score, and risk level analysis through an interactive Streamlit dashboard.

---

## 🚀 Features

* 🌦 Real-time weather data using OpenWeatherMap API
* 🤖 Machine Learning-based flood prediction
* 📊 Interactive Streamlit dashboard
* 📈 Flood probability estimation
* 🎯 Prediction confidence score
* 🚨 Risk level classification

  * Low Risk
  * Medium Risk
  * High Risk
  * Critical Risk
* 🏙 City-based prediction system
* 📱 Simple and user-friendly interface

---

## 🛠 Technologies Used

### Programming Language

* Python

### Libraries

* Streamlit
* Pandas
* NumPy
* Scikit-Learn
* Requests
* Pickle

### API

* OpenWeatherMap API

### Machine Learning

* Classification Algorithms (trained using flood dataset)

---

## 📂 Project Structure

```text
Project4E/
│
├── app.py
├── weather.py
├── train_model.py
├── model.pkl
├── Flood prediction System.csv
├── requirements.txt
├── projectFlood.ipynb
├── Flood Prediction.pptx
└── venv/
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/flood-prediction-system.git
cd flood-prediction-system
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 API Setup

1. Create an account on OpenWeatherMap.
2. Generate an API key.
3. Open `weather.py`.
4. Replace:

```python
API_KEY = "YOUR_API_KEY"
```

with your actual API key.

---

## ▶️ Run the Project

```bash
streamlit run app.py
```

After running, open:

```text
http://localhost:8501
```

---

## 📊 Working of the System

1. User enters a city name.
2. Weather data is fetched using OpenWeatherMap API.
3. Rainfall, humidity, and temperature are extracted.
4. Water level is estimated.
5. Data is passed to the trained Machine Learning model.
6. Flood probability is calculated.
7. Dashboard displays:

   * Flood Prediction
   * Probability Percentage
   * Confidence Score
   * Risk Level

---

## 🎯 Risk Levels

| Probability | Risk Level       |
| ----------- | ---------------- |
| 0% - 25%    | 🟢 Low Risk      |
| 25% - 50%   | 🟡 Medium Risk   |
| 50% - 75%   | 🔴 High Risk     |
| 75% - 100%  | 🟣 Critical Risk |

---

## 📸 Screenshots

Add screenshots of your dashboard here.

---

## 📈 Future Improvements

* Flood prediction history
* Weather forecast integration
* SMS/Email alerts
* Interactive maps
* Mobile application
* Cloud deployment

---

## 👨‍💻 Author

**Arpit**

Student Project – Flood Prediction and Detection System

---

## 📜 License

This project is developed for educational and academic purposes.
