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
        
        # Use gemini-2.5-flash for vision tasks
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Load the image
        img = Image.open(image_path)
        
        # Precise Hydrological Gauge Reader with Chain-of-Thought reasoning and Scene Validation
        prompt = """### PRECISE HYDROLOGICAL GAUGE READER

**ROLE**
You are an expert Hydrologist and Computer Vision Analyst specializing in reading vertical staff gauges in flood conditions. Your priority is PRECISION over estimation, and VALIDATION of the scene.

**CORE DIRECTIVE**
You must first validate the scene, then extract the water level reading from an image of a numbered gauge. You must use a "Chain of Thought" process to verify the direction of the scale and the relative position of the water line before generating a final number.

**ANALYSIS PROTOCOL (Follow these steps strictly)**

**STEP 0: SCENE DETECTION**
Before reading any numbers, identify the scene type:
- Is this a REAL water body with a physical gauge in water? (ideal)
- Is this a phone/screen showing a gauge photo? (allowed for demo, flag it)
- Is there NO gauge visible at all? (invalid)

**SCENE TYPES:**
- "real_scene": Physical gauge actually installed in water at a river/stream
- "phone_image": Image of a phone/screen displaying a gauge photo (ALLOWED FOR DEMO)
- "dry_gauge": Gauge visible but no water present
- "no_gauge": No gauge visible in the image
- "invalid": Indoor/irrelevant image

**NOTE:** Phone images are ALLOWED for demo purposes. If detected, set is_phone_image=true and still attempt to read the gauge.

1.  **Identify Visible Markers:**
    * List all clearly legible numbers on the gauge from Top to Bottom.
    * *Example Output:* "Visible numbers are 9, 8, 7, 6."

2.  **Determine Scale Direction:**
    * Do the numbers increase as you go UP or DOWN?
    * *Observation:* Standard flood gauges increase from bottom to top (e.g., 6 is below 7).

3.  **Locate the Water Line (Meniscus):**
    * Identify the exact pixel line where the gauge intersects the water surface.
    * **CRITICAL CHECK:** Is the water line *between* two visible numbers, or is it *below* the lowest visible number?
    * *Constraint:* If the lowest visible number is '6' and the water line is below that number (revealing tick marks under the 6), the value MUST be less than 6. Do not guess a value between 6 and 7.

4.  **Calculate & Refine:**
    * Count the minor graduation marks (ticks) between the main numbers to determine the scale (usually 0.1m or 0.02m per tick).
    * Interpolate the exact position of the water line based on these ticks.

5.  **Tamper Detection:**
    * Check for signs of image manipulation: unusual pixelation, inconsistent shadows, digital editing artifacts, unnatural colors.
    * Check if gauge numbers look digitally altered or pasted.

**FEW-SHOT EXAMPLES FOR LOGIC CORRECTION**

* **Bad Logic:** "I see 6 and 7. The gauge is clear. Reading is 6.5." (Incorrect because water was below 6).
* **Correct Logic:** "Visible numbers are 7 and 6. The scale increases upwards. The water line is below the 6 mark by approximately 2 minor ticks. Therefore, the reading is below 6.0. Estimated reading: 5.8m."

**RESPONSE FORMAT**
Return ONLY this JSON structure:
{
    "scene_type": "real_scene",
    "scene_valid": true,
    "is_phone_image": false,
    "scene_issue": null,
    "no_gauge_detected": false,
    "visible_markers": [7, 6, 5],
    "scale_direction": "increases upwards",
    "water_line_position": "below the 6 mark by 2 ticks",
    "reasoning": "Scene shows real river with gauge partially submerged. Visible numbers are 7, 6, 5. Scale increases upwards. Water line is below 6 mark by approximately 2 minor ticks (each tick = 0.1m). Reading = 6.0 - 0.2 = 5.8m",
    "water_level": 5.8,
    "confidence": 0.85,
    "is_valid": true,
    "gauge_location": "center of image, vertical staff gauge in water",
    "tamper_detected": false,
    "tamper_reason": null,
    "image_quality": "good",
    "suggestions": [],
    "reason": "Clear reading at 5.8m - water line is 2 ticks below the 6 mark"
}

**FIELD EXPLANATIONS:**
- scene_type: "real_scene", "phone_image", "dry_gauge", "no_gauge", or "invalid"
- scene_valid: true if gauge reading is possible (real_scene OR phone_image). false for no_gauge/invalid.
- is_phone_image: true if image is a phone/screen showing a gauge photo (ALLOWED FOR DEMO)
- scene_issue: Description if scene has issues
- no_gauge_detected: true if no gauge/staff visible in image at all
- visible_markers: Array of numbers visible on the gauge (from top to bottom)
- scale_direction: "increases upwards" or "increases downwards"
- water_line_position: Detailed description of where water meets gauge relative to markers
- reasoning: Your step-by-step chain of thought logic
- water_level: Final reading in METERS (decimal). Set to null if unreadable.
- confidence: 0.0 to 1.0 (High=0.8+, Medium=0.5-0.8, Low=<0.5). Lower by 0.1 if phone_image.
- is_valid: true if gauge is readable (even from phone image). false if no reading possible.
- gauge_location: Where in the image the gauge appears
- tamper_detected: true if image shows manipulation signs
- tamper_reason: Explanation if tampered
- image_quality: "excellent", "good", "fair", "poor", or "unusable"
- suggestions: Tips for better reading if confidence < 0.8
- reason: Brief summary of your reading

**CRITICAL RULES:**
1. Phone images ARE ALLOWED for demo - set is_phone_image=true and still read the gauge
2. ALWAYS verify scale direction before reading
3. If water is BELOW the lowest visible number, the reading MUST be less than that number
4. Count ticks carefully - don't estimate
5. If unsure, lower your confidence score and add suggestions
6. Return ONLY the JSON, no other text"""
        
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