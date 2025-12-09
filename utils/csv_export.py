"""
CSV Export Utility for Water Level Submissions
Auto-exports all submissions to CSV and updates on new submissions
"""

import csv
import os
from datetime import datetime


def export_submissions_to_csv(submissions, output_path='reports/water_level_submissions.csv'):
    """
    Export water level submissions to a CSV file.
    
    Args:
        submissions: List of WaterLevelSubmission objects
        output_path: Path to save the CSV file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Define CSV headers
    headers = [
        'ID',
        'Timestamp',
        'Site Name',
        'Site ID',
        'Water Level (m)',
        'Latitude',
        'Longitude',
        'Verification Method',
        'QR Code Scanned',
        'User',
        'Quality Rating',
        'Sync Status',
        'Tamper Score',
        'Notes',
        'Photo Filename',
        'Confidence Score'
    ]
    
    # Write to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for sub in submissions:
            row = [
                sub.id,
                sub.timestamp.strftime('%Y-%m-%d %H:%M:%S') if sub.timestamp else '',
                sub.site.name if sub.site else 'Unknown',
                sub.site_id,
                sub.water_level,
                sub.gps_latitude,
                sub.gps_longitude,
                sub.verification_method,
                sub.qr_code_scanned or '',
                sub.user.username if sub.user else 'Unknown',
                sub.quality_rating,
                sub.sync_status,
                sub.tamper_score,
                sub.notes or '',
                sub.photo_filename or '',
                getattr(sub, 'ai_confidence', '') or ''
            ]
            writer.writerow(row)
    
    print(f"✅ Exported {len(submissions)} submissions to {output_path}")
    return output_path


def append_submission_to_csv(submission, output_path='reports/water_level_submissions.csv'):
    """
    Append a single new submission to the CSV file.
    Creates the file with headers if it doesn't exist.
    
    Args:
        submission: WaterLevelSubmission object
        output_path: Path to the CSV file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    file_exists = os.path.exists(output_path)
    
    # Define headers (same as in export function)
    headers = [
        'ID',
        'Timestamp',
        'Site Name',
        'Site ID',
        'Water Level (m)',
        'Latitude',
        'Longitude',
        'Verification Method',
        'QR Code Scanned',
        'User',
        'Quality Rating',
        'Sync Status',
        'Tamper Score',
        'Notes',
        'Photo Filename',
        'Confidence Score'
    ]
    
    with open(output_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write headers if file is new
        if not file_exists:
            writer.writerow(headers)
        
        # Write the new submission
        row = [
            submission.id,
            submission.timestamp.strftime('%Y-%m-%d %H:%M:%S') if submission.timestamp else '',
            submission.site.name if submission.site else 'Unknown',
            submission.site_id,
            submission.water_level,
            submission.gps_latitude,
            submission.gps_longitude,
            submission.verification_method,
            submission.qr_code_scanned or '',
            submission.user.username if submission.user else 'Unknown',
            submission.quality_rating,
            submission.sync_status,
            submission.tamper_score,
            submission.notes or '',
            submission.photo_filename or '',
            getattr(submission, 'ai_confidence', '') or ''
        ]
        writer.writerow(row)
    
    print(f"✅ Appended submission #{submission.id} to {output_path}")
    return output_path


def get_csv_path():
    """Get the standard CSV file path."""
    return 'reports/water_level_submissions.csv'
