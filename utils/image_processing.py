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
        
        # Use gemini-2.5-flash-lite for vision tasks
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        # Load the image
        img = Image.open(image_path)
        
        # Comprehensive prompt for water gauge reading with tamper detection
        prompt = """You are an expert hydrologist and image analyst who specializes in reading water level gauges and detecting image tampering.

TASK: Analyze this image to read the water level gauge and check for any signs of tampering or manipulation.

## STEP 1: IMAGE QUALITY ASSESSMENT
First, evaluate the image quality:
- Is the image clear or blurry?
- Is the lighting adequate?
- Is the gauge visible and readable?
- Is the image taken from a suitable angle?

## STEP 2: GAUGE DETECTION
Look for the water level gauge in the image:
- Identify the measuring stick/staff with painted markings
- Note the position of the gauge (left/right/center of image)
- Check if numbers are visible (usually 0-5 meters or similar)
- Look for subdivision marks between main numbers

## STEP 3: WATER LEVEL READING
Find and read the water level:
- Locate where the water surface meets the gauge
- Read the exact measurement at the water line
- Note the units (meters or centimeters - convert to meters)
- Be precise with decimals (e.g., 2.35 meters)

## STEP 4: TAMPER DETECTION
Check for signs of image manipulation:
- Unusual pixelation or artifacts around gauge numbers
- Inconsistent lighting or shadows
- Signs of digital editing (clone stamping, airbrushing)
- Unnatural color patterns
- Gauge numbers that look altered or pasted
- Water line that looks artificially drawn
- Missing or inconsistent reflections
- Perspective distortions that don't match surroundings

## OUTPUT FORMAT
Return ONLY this JSON structure:
{
    "water_level": 2.35,
    "confidence": 0.85,
    "is_valid": true,
    "gauge_location": "center-left of image, clearly visible",
    "water_line_position": "water meets gauge at 2.35m mark",
    "tamper_detected": false,
    "tamper_reason": null,
    "image_quality": "good",
    "suggestions": [],
    "reason": "Clear reading at 2.35m mark on vertical gauge staff"
}

## FIELD EXPLANATIONS:
- water_level: Reading in METERS (decimal number, e.g., 2.35). Set to null if unreadable.
- confidence: 0.0 to 1.0 based on clarity and certainty
- is_valid: true if you can read the gauge, false otherwise
- gauge_location: Describe WHERE in the image the gauge is (e.g., "left side", "center", "not visible")
- water_line_position: Describe where water meets the gauge
- tamper_detected: true if image shows signs of manipulation
- tamper_reason: If tampered, explain what looks suspicious
- image_quality: "excellent", "good", "fair", "poor", or "unusable"
- suggestions: Array of tips to get better reading. Include if confidence < 0.8 or quality is poor. Examples:
  - "Move closer to the gauge for clearer numbers"
  - "Ensure better lighting on the gauge"
  - "Take photo straight-on, not at an angle"
  - "Avoid reflections on the water surface"
  - "Keep camera steady to avoid blur"
  - "Make sure the full gauge scale is visible"
- reason: Brief explanation of your reading or why it failed

## IMPORTANT NOTES:
- If image looks tampered, set tamper_detected to true and explain in tamper_reason
- If you cannot read the gauge, set water_level to null and is_valid to false
- Always provide helpful suggestions when confidence is low
- Be thorough but return ONLY the JSON, nothing else

Return ONLY the JSON object, no other text."""
        
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