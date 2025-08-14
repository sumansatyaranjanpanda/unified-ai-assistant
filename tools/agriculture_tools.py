# tools/agriculture_tools.py
from langchain.tools import tool

# ============ WEATHER TOOL ============
@tool("WeatherTool", return_direct=False)
def fetch_forecast(location: str, days: int = 3):
    """
    Fetch weather forecast for a given location and number of days.
    This is a stub implementation. Replace with a call to a weather API like OpenWeather.
    """
    # Example stub output
    forecast_data = {
        "location": location,
        "days": days,
        "forecast": [
            {"day": "2025-08-10", "temp": "30°C", "condition": "Sunny"},
            {"day": "2025-08-11", "temp": "28°C", "condition": "Light Rain"},
            {"day": "2025-08-12", "temp": "29°C", "condition": "Cloudy"}
        ]
    }
    return forecast_data

# ============ PEST DETECTION TOOL ============
@tool("PestDetectTool")
def analyze_image(image_path: str):
    """
    Analyze plant or crop image to detect possible pest or disease.
    Stub: Replace with vision model inference (e.g., YOLOv8, ResNet, etc.).
    """
    # Simulate pest detection
    pest_diagnosis = {
        "image": image_path,
        "detected_pest": "Aphids",
        "confidence": 0.92,
        "recommendation": "Use neem oil spray every 3 days for 2 weeks."
    }
    return pest_diagnosis

# ============ FERTILIZER TOOL ============
@tool("FertilizerTool")
def recommend_fertilizer(crop: str, soil: str, stage: str):
    """
    Recommend fertilizer based on crop type, soil, and growth stage.
    """
    recommendation = f"For {crop} in {soil} soil at {stage} stage, use NPK 10:26:26 at 50kg/acre."
    return recommendation

# ============ MARKET PRICE TOOL ============
@tool("MarketPriceTool")
def get_prices(crop: str, location: str):
    """
    Get latest market prices for the given crop and location.
    Stub: Replace with real data from an API like AgMarkNet.
    """
    price_info = {
        "crop": crop,
        "location": location,
        "price_per_kg": 24.5,
        "last_updated": "2025-08-10"
    }
    return price_info

# ============ RAG TOOL ============
@tool("RAGTool")
def retrieval_qa(question: str):
    """
    Answer crop-related questions using RAG from agriculture docs.
    This function loads a retriever + LLM pipeline from agriculture_qa.py
    """
    from agriculture_qa import get_agriculture_qa
    qa = get_agriculture_qa()
    return qa.run(question)

# ============ IRRIGATION SCHEDULER TOOL ============
@tool("SchedulerTool")
def irrigation_schedule(soil: str, forecast: dict):
    """
    Suggest irrigation schedule based on soil type and upcoming weather.
    """
    schedule = f"For {soil} soil, irrigate every 3 days unless rainfall > 5mm is expected in forecast."
    return schedule

# ============ GOVERNMENT SCHEME TOOL ============
@tool("GovSchemeTool")
def find_schemes(profile: dict):
    """
    Find government schemes based on farmer profile (location, land size, crop).
    Stub: Replace with actual scraping/API.
    """
    schemes_list = [
        {"name": "PM Kisan Samman Nidhi", "benefit": "₹6,000/year direct benefit"},
        {"name": "Soil Health Card Scheme", "benefit": "Free soil testing every 3 years"},
        {"name": "PM Fasal Bima Yojana", "benefit": "Crop insurance against natural calamities"}
    ]
    return schemes_list
