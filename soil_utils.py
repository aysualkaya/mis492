import httpx
import logging
import json

logger = logging.getLogger(__name__)

# Soil type encoder mapping
SOIL_TYPE_ENCODER = {
    "Black": 0,
    "Clay": 1,
    "Loamy": 2,
    "Peaty": 3,
    "Red": 4,
    "Saline": 5,
    "Sandy": 6,
    "Unknown": 7
}

def encode_soil_type(label, encoder=None):
    """Encode soil type label to numeric value"""
    if encoder is None:
        encoder = SOIL_TYPE_ENCODER
    return encoder.get(label, encoder["Unknown"])

def get_partial_soil_data(lat, lon):
    """
    Fetch soil data from SoilGrids API
    Args:
        lat: Latitude (North-South position)
        lon: Longitude (East-West position)
    """
    base_url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    properties = ["phh2o", "nitrogen", "clay", "sand", "silt"]

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://rest.isric.org/soilgrids/v2.0/docs",
        "Origin": "https://rest.isric.org"
    }

    # Log the coordinates being used
    logger.info(f"🌍 Fetching soil data for coordinates: lat={lat}, lon={lon}")
    
    params = {
        "lon": lon,
        "lat": lat,
        "property": properties,
        "depth": "0-5cm",
        "value": "mean"
    }
    
    try:
        response = httpx.get(base_url, params=params, headers=headers, timeout=30)
        logger.info(f"📡 API URL: {response.url}")
        logger.info(f"📄 Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Debug: Log the raw response structure
            logger.debug(f"📊 Raw response keys: {list(data.keys())}")
            if "properties" in data:
                logger.debug(f"Properties keys: {list(data['properties'].keys())}")
                if "layers" in data["properties"]:
                    logger.debug(f"Number of layers: {len(data['properties']['layers'])}")
            
            soil_data = {}
            
            # Check if we have the expected structure
            if "properties" not in data:
                logger.error("❌ No 'properties' key in API response")
                return None
                
            if "layers" not in data["properties"]:
                logger.error("❌ No 'layers' key in properties")
                return None
            
            layers = data["properties"]["layers"]
            if not layers:
                logger.error("❌ Empty layers array")
                return None
            
            # Process each property layer
            for layer in layers:
                prop_name = layer.get("name")
                if not prop_name:
                    logger.warning("⚠️ Layer without name found")
                    continue
                    
                logger.debug(f"Processing layer: {prop_name}")
                
                if prop_name in properties:
                    # Check if we have depths data
                    if "depths" not in layer:
                        logger.warning(f"⚠️ No depths data for {prop_name}")
                        continue
                        
                    depths_data = layer.get("depths", [])
                    if not depths_data:
                        logger.warning(f"⚠️ Empty depths array for {prop_name}")
                        continue
                    
                    # Find the 0-5cm depth data
                    target_depth = None
                    for depth_info in depths_data:
                        depth_label = depth_info.get("label", "")
                        logger.debug(f"  Found depth: {depth_label}")
                        if depth_label == "0-5cm":
                            target_depth = depth_info
                            break
                    
                    if target_depth:
                        # Check if values exist
                        if "values" not in target_depth:
                            logger.warning(f"⚠️ No values section for {prop_name} at 0-5cm")
                            continue
                            
                        values = target_depth.get("values", {})
                        mean_value = values.get("mean")
                        
                        if mean_value is not None:
                            # Apply unit conversion
                            unit_measure = layer.get("unit_measure", {})
                            d_factor = unit_measure.get("d_factor", 1)
                            
                            # Convert based on d_factor
                            converted_value = mean_value / d_factor
                            soil_data[prop_name] = converted_value
                            logger.info(f"✅ {prop_name.upper()}: {converted_value} (raw: {mean_value}, d_factor: {d_factor})")
                        else:
                            logger.warning(f"⚠️ No mean value for {prop_name.upper()}")
                            logger.debug(f"  Available values: {list(values.keys())}")
                    else:
                        logger.warning(f"⚠️ No 0-5cm depth data for {prop_name.upper()}")
                        # Log available depths for debugging
                        available_depths = [d.get("label", "unknown") for d in depths_data]
                        logger.debug(f"  Available depths: {available_depths}")
            
            if not soil_data:
                logger.warning("⚠️ No soil data extracted from API response")
                # Log the full response for debugging
                logger.debug(f"Full response for debugging: {json.dumps(data, indent=2)}")
                return None
                
        else:
            logger.warning(f"⚠️ API request failed with status {response.status_code}")
            logger.debug(f"Response content: {response.text[:500]}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Exception fetching soil data: {e}")
        return None

    # Convert to expected format for your model
    processed_data = {
        "ph": soil_data.get("phh2o", 7.0),  # pH is already converted by d_factor
        "n": soil_data.get("nitrogen", 1.0),  # Nitrogen in g/kg
        "p": 20.0,  # Default P value (not available from SoilGrids)
        "k": 200.0,  # Default K value (not available from SoilGrids)
        "clay_percent": soil_data.get("clay", 20.0),
        "sand_percent": soil_data.get("sand", 40.0),
        "silt_percent": soil_data.get("silt", 40.0)
    }
    
    logger.info(f"✅ Processed soil data: {processed_data}")
    return processed_data

def map_texture_to_soil_type(clay, sand, silt):
    """Map texture data to soil type"""
    if clay is None or sand is None:
        return "Unknown"
    
    if clay > 40:
        return "Clay"
    elif sand > 60:
        return "Sandy"
    elif silt > 40:
        return "Loamy"
    elif clay > 20:
        return "Loamy"
    else:
        return "Unknown"

def validate_soil_data(data):
    """Ensure all required soil keys are present and valid"""
    if data is None:
        return False
    required_keys = ['ph', 'n', 'p', 'k']
    return all(k in data and isinstance(data[k], (int, float)) for k in required_keys)