from PIL import Image, ImageDraw, ImageFont
import os
from datetime import datetime

def add_timestamp_to_image(image_path, timestamp, latitude, longitude):
    """
    Add timestamp and location metadata overlay to the captured image
    """
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            draw = ImageDraw.Draw(img)
            
            # Try to use a font, fallback to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 30)
            except:
                font = ImageFont.load_default()
            
            # Format the text
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            location_str = f"Lat: {latitude:.6f}, Lon: {longitude:.6f}"
            
            # Add text background for better readability
            text = f"{timestamp_str}\n{location_str}"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Draw background rectangle
            margin = 10
            draw.rectangle([
                margin, 
                margin, 
                margin + text_width + 10, 
                margin + text_height + 10
            ], fill='black')
            
            # Draw text
            draw.text((margin + 5, margin + 5), text, fill='white', font=font)
            
            # Save the modified image
            img.save(image_path)
            return True
            
    except Exception as e:
        print(f"Error adding timestamp to image: {e}")
        return False

def analyze_water_gauge(image_path):
    """
    Analyze the image using Gemini API to read the water level gauge.
    Returns a dictionary:
    {
        "water_level": float or None,
        "confidence": float (0.0 to 1.0),
        "is_valid": bool,
        "reason": str
    }
    """
    import google.generativeai as genai
    import json
    
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print("GOOGLE_API_KEY not found in environment variables.")
        return None
        
    try:
        genai.configure(api_key=api_key)
        
        # Use gemini-1.5-flash for vision tasks
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Load the image
        img = Image.open(image_path)
        
        # Improved prompt for water gauge reading - MORE DETAILED
        prompt = """You are an expert hydrologist who specializes in reading water level gauges accurately.

TASK: Read the water level from this gauge image with precision.

INSTRUCTIONS:
1. Look carefully at the measuring stick/gauge in the image
2. Find where the water surface meets the gauge
3. Read the exact number marked at the water line
4. The gauge usually has markings in meters or centimeters

IMPORTANT GUIDELINES:
- Look for painted numbers on the gauge staff (e.g., 1, 2, 3, 4, 5 meters)
- Look for smaller subdivision marks between the main numbers
- The water level is where the water surface intersects the gauge
- If gauge shows cm, convert to meters (e.g., 250 cm = 2.50 meters)
- Be precise - include decimals (e.g., 2.35 meters, not just 2 meters)

OUTPUT: Return ONLY this JSON format with your reading:
{
    "water_level": 2.35,
    "confidence": 0.85,
    "is_valid": true,
    "reason": "Clear reading at 2.35m mark on gauge"
}

NOTES:
- water_level should be a decimal number in METERS (e.g., 2.35)
- confidence should be between 0.0 and 1.0
- is_valid should be true if you can see and read the gauge
- reason should explain what you see

Return ONLY the JSON, nothing else."""
        
        response = model.generate_content([prompt, img])
        text = response.text.strip()
        
        # Clean up potential markdown code blocks
        if text.startswith('```json'):
            text = text[7:]
        elif text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        text = text.strip()
        
        print(f"Gemini AI Response: {text}")
        
        try:
            result = json.loads(text)
            # Ensure we have all required fields
            if 'water_level' not in result:
                result['water_level'] = None
            if 'confidence' not in result:
                result['confidence'] = 0.5
            if 'is_valid' not in result:
                result['is_valid'] = result.get('water_level') is not None
            if 'reason' not in result:
                result['reason'] = 'Analyzed with OpenCV'
            return result
        except json.JSONDecodeError as e:
            print(f"Could not parse AI response as JSON: {text}, error: {e}")
            return None
            
    except Exception as e:
        print(f"Gemini API Error: {e}")
        import traceback
        traceback.print_exc()
        return None