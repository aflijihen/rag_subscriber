import requests

def get_weather_data(latitude, longitude):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=temperature_2m,precipitation"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception if the response status code is not 200
        weather_data = response.json()  # Parse the JSON response
        return weather_data
    except requests.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

if __name__ == "__main__":
    tunis_latitude = 36.8065
    tunis_longitude = 10.1815
    weather_data = get_weather_data(tunis_latitude, tunis_longitude)
    if weather_data:
        print("Weather data for Tunis:")
        print(f"Temperature at 2m: {weather_data['hourly']['temperature_2m'][0]}°C")
        print(f"Precipitation: {weather_data['hourly']['precipitation'][0]} mm")
    else:
        print("Unable to fetch weather data.")
