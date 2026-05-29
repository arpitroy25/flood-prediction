import requests

API_KEY = "e83c243882bd53930307a360b39c0385"

def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    response = requests.get(url)

    print("Status Code:", response.status_code)
    print("Response:", response.text)

    if response.status_code == 200:
        data = response.json()

        rainfall = data.get("rain", {}).get("1h", 0)
        humidity = data["main"]["humidity"]
        temperature = data["main"]["temp"]

        return {
            "rainfall": rainfall,
            "humidity": humidity,
            "temperature": temperature
        }

    return None
