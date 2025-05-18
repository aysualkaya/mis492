import ee
import requests
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
import datetime

load_dotenv()
ee.Initialize(project='agromind-b2196')

# Soil Type Label Encoder (Global)
SOIL_TYPES = ["Black", "Red", "Peaty", "Saline", "Sandy", "Clay", "Loamy", "Silty", "Unknown"]
SOIL_TYPE_ENCODER = LabelEncoder()
SOIL_TYPE_ENCODER.fit(SOIL_TYPES)

def get_location():
    try:
        r = requests.get("https://ipinfo.io/json")
        lat, lon = map(float, r.json()["loc"].split(","))
        return lat, lon
    except Exception as e:
        print(f"⚠️ Failed to get location via IP: {e}")
        return 0.0, 0.0

def get_location_details(lat, lon):
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
    return k - 273.15

def dewpoint_to_humidity(temp_c, dewpoint_c):
    return max(0, min(100, 100 * (112 - 0.1 * temp_c + dewpoint_c) / (112 + 0.9 * temp_c)))

def get_weighted_climate(lat, lon, target_month=0):
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

def fetch_soil_with_fallback(lat, lon, max_radius=0.5, step=0.1):
    try:
        # Ana konumdan veri al
        soil_data = get_soil_data(lat, lon)
        if soil_data:
            print(f"✅ Soil data found at ({lat}, {lon}): {soil_data}")
            return soil_data

        # Yakın çevrede arama yap
        for radius in [step * i for i in range(1, int(max_radius / step) + 1)]:
            offsets = [(radius, 0), (-radius, 0), (0, radius), (0, -radius),
                       (radius, radius), (-radius, -radius), (radius, -radius), (-radius, radius)]

            for lat_offset, lon_offset in offsets:
                nearby_data = get_soil_data(lat + lat_offset, lon + lon_offset)
                if nearby_data:
                    print(f"✅ Soil data found at ({lat + lat_offset}, {lon + lon_offset}): {nearby_data}")
                    return nearby_data

        # Hiçbir veri bulunamazsa None döndür
        print(f"⚠️ No soil data found for ({lat}, {lon}) or nearby areas.")
        return None

    except Exception as e:
        print(f"❌ Error fetching soil data: {e}")
        return None

def map_texture_to_soil_type(texture_class, clay, sand, silt):
    if texture_class != "Unknown":
        return texture_class.capitalize()
    elif clay > 40:
        return "Clay"
    elif sand > 70:
        return "Sandy"
    elif silt > 40:
        return "Silty"
    elif 20 < clay <= 40 and 20 < sand <= 70 and 20 < silt <= 70:
        return "Loamy"
    elif clay > 20 and sand < 20 and silt < 20:
        return "Black"
    elif sand > 60 and clay < 10:
        return "Red"
    elif silt > 50 and clay < 10 and sand < 20:
        return "Peaty"
    elif sand > 20 and silt > 20 and clay < 10:
        return "Saline"
    else:
        return "Unknown"
