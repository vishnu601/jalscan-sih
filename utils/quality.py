import os
from utils.geofence import calculate_distance

def calculate_quality_score(submission_data, site=None, ai_result=None):
    """
    Calculates a quality score (1-5) for a submission based on available data.
    
    Args:
        submission_data (dict): Contains 'latitude', 'longitude', 'water_level', 'photo_path'
        site (MonitoringSite): The site object (optional)
        ai_result (dict): Result from AI analysis containing 'water_level' and 'confidence' (optional)
        
    Returns:
        int: Quality score between 1 and 5
        list: List of reasons for deduction (for debugging or display)
    """
    score = 5
    deductions = []

    # 1. GPS Verification (if site is provided)
    if site and 'latitude' in submission_data and 'longitude' in submission_data:
        try:
            distance = calculate_distance(
                float(submission_data['latitude']), float(submission_data['longitude']),
                site.latitude, site.longitude
            )
            
            if distance > 200:
                score -= 2
                deductions.append(f"GPS Offset > 200m ({int(distance)}m)")
            elif distance > 50:
                score -= 1
                deductions.append(f"GPS Offset > 50m ({int(distance)}m)")
        except Exception as e:
            # If GPS calc fails, minor penalty or ignore
            pass
    
    # 2. AI Verification Consistency
    if ai_result:
        ai_level = ai_result.get('water_level')
        confidence = ai_result.get('confidence', 0)
        
        # Check consistency between manual and AI reading
        if ai_level is not None and 'water_level' in submission_data:
            try:
                manual_level = float(submission_data['water_level'])
                diff = abs(manual_level - float(ai_level))
                
                if diff > 0.5:
                    score -= 2
                    deductions.append(f"Major AI Discrepancy ({diff:.2f}m)")
                elif diff > 0.1:
                    score -= 1
                    deductions.append(f"Minor AI Discrepancy ({diff:.2f}m)")
            except ValueError:
                pass

        # Check AI Confidence (Image Clarity proxy)
        if confidence < 0.5:
            score -= 1
            deductions.append(f"Low AI Confidence ({int(confidence*100)}%)")
            
    # 3. Image Quality (File Size Check)
    if 'photo_path' in submission_data and submission_data['photo_path']:
        try:
            if os.path.exists(submission_data['photo_path']):
                size_kb = os.path.getsize(submission_data['photo_path']) / 1024
                if size_kb < 50:
                    score -= 1
                    deductions.append("Low Resolution Image (< 50KB)")
        except Exception:
            pass
            
    # Ensure score is within bounds
    final_score = max(1, min(5, score))
    
    return final_score, deductions
