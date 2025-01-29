import pandas as pd
import requests
from datetime import datetime
import difflib
from typing import Dict, Tuple, List

class IndianLocations:
    def __init__(self):
        # Load the CSV dataset
        try:
            # Explicitly specify engine as 'python' for CSV reading
            self.df = pd.read_csv(r"C:\sreejha_project\project\models\Indian_Cities_Database.xlsx.csv")
            
            # Create a dictionary of cities with their coordinates
            self.cities = {}
            for _, row in self.df.iterrows():
                # Convert coordinates to float to ensure they're numeric
                try:
                    lat = float(row['Lat'])
                    long = float(row['Long'])
                    self.cities[str(row['City']).lower()] = {
                        'coords': (lat, long),
                        'state': str(row['State'])
                    }
                except (ValueError, TypeError):
                    continue  # Skip rows with invalid coordinates
                    
            print(f"Successfully loaded {len(self.cities)} cities from the dataset.")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Creating sample dataset...")
            # Sample data in case file is not found
            self.cities = {
                'delhi': {'coords': (28.6139, 77.2090), 'state': 'Delhi'},
                'mumbai': {'coords': (19.0760, 72.8777), 'state': 'Maharashtra'},
                'kolkata': {'coords': (22.5726, 88.3639), 'state': 'West Bengal'}
            }

    def search_location(self, query: str) -> Tuple[str, Dict]:
        """Search for a location using fuzzy matching."""
        query = query.lower().strip()
        
        # Print available cities for debugging
        if query == "list":
            print("\nAvailable cities:")
            for city, info in self.cities.items():
                print(f"- {city.title()} ({info['state']})")
            return None, None
        
        # Direct match
        if query in self.cities:
            return query, self.cities[query]
        
        # Fuzzy matching
        matches = difflib.get_close_matches(query, self.cities.keys(), n=1, cutoff=0.6)
        if matches:
            return matches[0], self.cities[matches[0]]
            
        return None, None

# [Previous WeatherMonitor class remains the same]
class WeatherMonitor:
    def __init__(self, location_system):
        """Initialize WeatherMonitor with location system."""
        self.location_system = location_system
        self.api_url = "https://api.open-meteo.com/v1/forecast"
        
    def get_weather_data(self, lat: float, lon: float) -> dict:
        """Fetch weather data from the API for given coordinates."""
        try:
            params = {
                'latitude': lat,
                'longitude': lon,
                'current': ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m'],
                'hourly': ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m'],
                'timezone': 'auto'
            }
            
            response = requests.get(self.api_url, params=params)
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
            
    def generate_alerts(self, weather_data: dict) -> List[str]:
        """Generate weather alerts based on current conditions."""
        alerts = []
        current = weather_data['current']
        
        # Temperature alerts
        temp = current['temperature_2m']
        if temp > 35:
            alerts.append("⚠️ HIGH TEMPERATURE ALERT: Temperature above 35°C may stress rice crops")
        elif temp < 15:
            alerts.append("⚠️ LOW TEMPERATURE ALERT: Temperature below 15°C may affect rice growth")
            
        # Humidity alerts
        humidity = current['relative_humidity_2m']
        if humidity > 85:
            alerts.append("⚠️ HIGH HUMIDITY ALERT: Increased risk of fungal diseases")
        elif humidity < 40:
            alerts.append("⚠️ LOW HUMIDITY ALERT: Risk of water stress in rice plants")
            
        # Wind speed alerts
        wind_speed = current['wind_speed_10m']
        if wind_speed > 30:
            alerts.append("⚠️ HIGH WIND ALERT: Strong winds may damage rice plants")
            
        return alerts
        
    def generate_recommendations(self, weather_data: dict) -> List[str]:
        """Generate recommendations based on weather conditions."""
        recommendations = []
        current = weather_data['current']
        
        # Temperature-based recommendations
        temp = current['temperature_2m']
        if temp > 35:
            recommendations.append("• Consider additional irrigation to combat heat stress")
            recommendations.append("• Apply mulch to help maintain soil moisture")
        elif temp < 15:
            recommendations.append("• Monitor crop closely for cold damage")
            recommendations.append("• Consider using cold-protective measures if available")
            
        # Humidity-based recommendations
        humidity = current['relative_humidity_2m']
        if humidity > 85:
            recommendations.append("• Monitor for signs of fungal diseases")
            recommendations.append("• Consider fungicide application if necessary")
        elif humidity < 40:
            recommendations.append("• Increase irrigation frequency")
            recommendations.append("• Consider using moisture retention techniques")
            
        # Wind-based recommendations
        wind_speed = current['wind_speed_10m']
        if wind_speed > 30:
            recommendations.append("• Check field for wind damage")
            recommendations.append("• Consider temporary wind barriers if possible")
            
        # Add general recommendation if no specific ones are needed
        if not recommendations:
            recommendations.append("• Current conditions are favorable for rice cultivation")
            recommendations.append("• Continue regular monitoring and maintenance")
            
        return recommendations

def main():
    print("\n=== Rice Crop Weather Monitoring System ===")
    location_system = IndianLocations()
    monitor = WeatherMonitor(location_system)
    
    while True:
        print("\nEnter your city name (or 'list' to see all cities, 'quit' to exit)")
        query = input("City: ").strip()
        
        if query.lower() == 'quit':
            print("Thank you for using the monitoring system. Goodbye!")
            break
            
        if query.lower() == 'list':
            location_system.search_location('list')
            continue
        
        city, location_info = location_system.search_location(query)
        
        if not city:
            print("City not found. Please check the spelling and try again.")
            print("You can type 'list' to see all available cities.")
            continue
            
        print(f"\nFetching weather data for {city.title()}, {location_info['state']}...")
        weather_data = monitor.get_weather_data(*location_info['coords'])
        
        if weather_data:
            current = weather_data['current']
            print(f"\n=== Current Weather Conditions ===")
            print(f"Time: {datetime.fromisoformat(current['time']).strftime('%Y-%m-%d %H:%M')}")
            print(f"Temperature: {current['temperature_2m']}°C")
            print(f"Humidity: {current['relative_humidity_2m']}%")
            print(f"Wind Speed: {current['wind_speed_10m']} km/h")
            
            # Get next 2 hours forecast
            print(f"\n=== 2-Hour Forecast ===")
            for i in range(2):
                hour_data = {
                    'time': weather_data['hourly']['time'][i],
                    'temp': weather_data['hourly']['temperature_2m'][i],
                    'humidity': weather_data['hourly']['relative_humidity_2m'][i],
                    'wind': weather_data['hourly']['wind_speed_10m'][i]
                }
                print(f"\nTime: {datetime.fromisoformat(hour_data['time']).strftime('%H:%M')}")
                print(f"Temperature: {hour_data['temp']}°C")
                print(f"Humidity: {hour_data['humidity']}%")
                print(f"Wind Speed: {hour_data['wind']} km/h")
            
            print("\n=== Alerts ===")
            alerts = monitor.generate_alerts(weather_data)
            if alerts:
                for alert in alerts:
                    print(alert)
            else:
                print("No critical alerts at this time.")
                
            print("\n=== Recommendations ===")
            recs = monitor.generate_recommendations(weather_data)
            for rec in recs:
                print(rec)
        else:
            print("Unable to fetch weather data. Please try again later.")

if __name__ == "__main__":
    main()