import ee
import requests
import numpy as np
import os
from dotenv import load_dotenv
import datetime

load_dotenv()
ee.Initialize(project='agromind-b2196')

def get_location():
    """Get current location using IP geolocation"""
    try:
        r = requests.get("https://ipinfo.io/json")
        lat, lon = map(float, r.json()["loc"].split(","))
        return lat, lon
    except Exception as e:
        print(f"⚠️ Failed to get location via IP: {e}")
        return 0.0, 0.0

def get_location_details(lat, lon):
    """Get human-readable location details from coordinates"""
    url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        address = data.get('address', {})
        city = address.get('city') or address.get('town') or address.get('village') or "Unknown City"
        country = address.get('country') or "Unknown Country"
        return f"{city}, {country}"
    except Exception as e:
        print(f"🌐 Location detail fetch failed: {e}")
        return "Unknown Location"

def kelvin_to_celsius(k):
    """Convert Kelvin to Celsius"""
    return k - 273.15

def dewpoint_to_humidity(temp_c, dewpoint_c):
    """Calculate relative humidity from temperature and dewpoint"""
    return max(0, min(100, 100 * (112 - 0.1 * temp_c + dewpoint_c) / (112 + 0.9 * temp_c)))

def get_weighted_climate(lat, lon, target_month=0):
    """
    Get weighted climate data for a specific location and month.
    Recent years are weighted more heavily than older years.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude  
        target_month (int): Target month (1-12), if 0 uses current month
        
    Returns:
        dict: Temperature and humidity data
    """
    point = ee.Geometry.Point(lon, lat)
    years = list(range(2000, 2025))
    weights = [0.1 + 0.9 * ((y - 2000) / (2024 - 2000)) for y in years]

    temp_sum, dew_sum, total_weight = 0, 0, 0

    current_month = datetime.datetime.now().month
    target_month = current_month if target_month == 0 else target_month

    for y, w in zip(years, weights):
        start_date = f"{y}-{target_month:02d}-01"
        end_date = f"{y+1}-01-01" if target_month == 12 else f"{y}-{target_month+1:02d}-01"
        ic = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").filterDate(start_date, end_date).filterBounds(point)
        temp_img = ic.select("temperature_2m").mean()
        dew_img = ic.select("dewpoint_temperature_2m").mean()

        try:
            temp_val = temp_img.reduceRegion(ee.Reducer.mean(), point.buffer(5000), 1000).get('temperature_2m').getInfo()
            dew_val = dew_img.reduceRegion(ee.Reducer.mean(), point.buffer(5000), 1000).get('dewpoint_temperature_2m').getInfo()
            if temp_val is not None and dew_val is not None:
                temp_sum += temp_val * w
                dew_sum += dew_val * w
                total_weight += w
        except Exception as e:
            print(f"Warning: Missing data for {y}-{target_month:02d}: {e}")

    if total_weight == 0:
        raise ValueError(f"No valid climate data found for the given month ({target_month}) and location.")

    temp_c = kelvin_to_celsius(temp_sum / total_weight)
    dew_c = kelvin_to_celsius(dew_sum / total_weight)
    humidity = dewpoint_to_humidity(temp_c, dew_c)

    return {"temperature": round(temp_c, 2), "humidity": round(humidity, 2)}