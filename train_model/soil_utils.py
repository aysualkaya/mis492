import requests 
import logging
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# SoilGrids API Configuration
SOILGRIDS_API_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
HEADERS = {"Accept": "application/json"}
PROPERTIES = {
    "ph": "phh2o",
    "n": "nitrogen", 
    "p": "phosphorus",
    "k": "potassium",
    "clay": "clay",
    "sand": "sand",
    "silt": "silt"
}
DEFAULT_DEPTH = "0-5cm"

# Global Soil Type Configuration
SOIL_TYPES = ["Black", "Red", "Peaty", "Saline", "Sandy", "Clay", "Loamy", "Silty", "Unknown"]
SOIL_TYPE_ENCODER = LabelEncoder()
SOIL_TYPE_ENCODER.fit(SOIL_TYPES)

def get_soil_data(lat, lon):
    """
    Fetch soil data from SoilGrids API for given coordinates.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        
    Returns:
        dict: Soil properties or None if no data available
    """
    try:
        url = f"{SOILGRIDS_API_URL}?lon={lon}&lat={lat}&property={'&property='.join(PROPERTIES.values())}&depth={DEFAULT_DEPTH}"
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        data = response.json()

        result = {}
        for key, soilgrid_key in PROPERTIES.items():
            try:
                value = data["properties"][soilgrid_key]["mean"]
                if isinstance(value, list):
                    result[key] = float(value[0])
                else:
                    result[key] = float(value)
            except Exception:
                logger.warning(f"Missing or malformed data for {key}")
                result[key] = None

        return result if any(v is not None for v in result.values()) else None

    except Exception as e:
        logger.error(f"SoilGrids API error: {e}")
        return None

def get_soil_data_with_fallback(lat, lon, max_radius=0.5, step=0.1):
    """
    Fetch soil data with fallback to nearby locations if no data found.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        max_radius (float): Maximum search radius in degrees
        step (float): Step size for expanding search
        
    Returns:
        dict: Soil properties or None if no data found
    """
    try:
        # Try exact location first
        soil_data = get_soil_data(lat, lon)
        if soil_data:
            logger.info(f"✅ Soil data found at ({lat}, {lon}): {soil_data}")
            return soil_data

        # Expand search in nearby areas
        for radius in [step * i for i in range(1, int(max_radius / step) + 1)]:
            offsets = [
                (radius, 0), (-radius, 0), (0, radius), (0, -radius),
                (radius, radius), (-radius, -radius), 
                (radius, -radius), (-radius, radius)
            ]

            for lat_offset, lon_offset in offsets:
                nearby_data = get_soil_data(lat + lat_offset, lon + lon_offset)
                if nearby_data:
                    logger.info(f"✅ Soil data found at ({lat + lat_offset}, {lon + lon_offset}): {nearby_data}")
                    return nearby_data

        logger.warning(f"⚠️ No soil data found for ({lat}, {lon}) or nearby areas.")
        return None

    except Exception as e:
        logger.error(f"❌ Error fetching soil data: {e}")
        return None

def map_texture_to_soil_type(clay, sand, silt):
    """
    Map soil texture components to soil type classification.
    
    Args:
        clay (float): Clay percentage
        sand (float): Sand percentage  
        silt (float): Silt percentage
        
    Returns:
        str: Soil type classification
    """
    if None in (clay, sand, silt):
        return "Unknown"
    
    # Clay-dominant soils
    if clay > 40:
        return "Clay"
    # Sand-dominant soils  
    elif sand > 70:
        return "Sandy"
    # Silt-dominant soils
    elif silt > 40:
        return "Silty"
    # Balanced loamy soils
    elif 20 < clay < 35 and 20 < sand < 70 and 20 < silt < 70:
        return "Loamy"
    # Organic-rich soils
    elif clay < 20 and sand < 52 and silt > 28:
        return "Peaty"
    # High clay with specific minerals
    elif clay > 20 and sand < 20 and silt < 20:
        return "Black"
    # Iron-rich sandy soils
    elif sand > 60 and clay < 10:
        return "Red"
    # Salt-affected soils
    elif sand > 20 and silt > 20 and clay < 10:
        return "Saline"
    else:
        return "Unknown"

def encode_soil_type(label, encoder=None):
    """
    Encode soil type label to numerical value.
    
    Args:
        label (str): Soil type label
        encoder (LabelEncoder): Custom encoder, uses global if None
        
    Returns:
        int: Encoded soil type value
    """
    if encoder is None:
        encoder = SOIL_TYPE_ENCODER
        
    try:
        return int(encoder.transform([label])[0])
    except:
        return int(encoder.transform(["Unknown"])[0])