#!/usr/bin/env python3
"""
Generate Mock Data for River Memory AI Demo
Creates realistic time-series data for demonstration purposes
"""

import os
import sys
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_mock_sites(db_session, num_sites: int = 5) -> List[Dict]:
    """Generate mock river monitoring sites"""
    from models import MonitoringSite
    
    site_data = [
        {"name": "Ganga River - Varanasi", "lat": 25.3176, "lon": 82.9739},
        {"name": "Yamuna River - Delhi", "lat": 28.6139, "lon": 77.2090},
        {"name": "Brahmaputra - Guwahati", "lat": 26.1445, "lon": 91.7362},
        {"name": "Godavari - Nashik", "lat": 19.9975, "lon": 73.7898},
        {"name": "Krishna River - Vijayawada", "lat": 16.5062, "lon": 80.6480},
        {"name": "Narmada - Jabalpur", "lat": 23.1815, "lon": 79.9864},
        {"name": "Tapi River - Surat", "lat": 21.1702, "lon": 72.8311},
        {"name": "Mahanadi - Cuttack", "lat": 20.4625, "lon": 85.8830},
    ]
    
    created_sites = []
    
    for i in range(min(num_sites, len(site_data))):
        data = site_data[i]
        
        # Check if site exists
        existing = MonitoringSite.query.filter_by(name=data["name"]).first()
        if existing:
            created_sites.append({"id": existing.id, **data})
            continue
        
        site = MonitoringSite(
            name=data["name"],
            latitude=data["lat"],
            longitude=data["lon"],
            river_code=f"RIVER{i+1:03d}",
            is_active=True,
            description=f"Mock monitoring site for {data['name']}"
        )
        db_session.add(site)
        db_session.commit()
        
        created_sites.append({"id": site.id, **data})
        print(f"Created site: {site.name} (ID: {site.id})")
    
    return created_sites


def generate_mock_analyses(
    db_session, 
    site_id: int, 
    days: int = 30,
    readings_per_day: int = 4
) -> int:
    """Generate mock River Memory analysis data for a site"""
    from models import RiverAnalysis
    
    now = datetime.utcnow()
    start_time = now - timedelta(days=days)
    
    # Base values with realistic variation
    base_level = random.uniform(150, 350)
    base_trend = random.choice([-0.5, 0, 0.5, 1.0])  # Rising/falling trend
    
    created_count = 0
    current_time = start_time
    
    # Simulate seasonal and daily patterns
    while current_time < now:
        # Daily pattern (higher during monsoon hours)
        hour = current_time.hour
        daily_factor = 1.0 + 0.1 * np.sin(hour * np.pi / 12)
        
        # Weekly pattern (slight weekend variation)
        day_of_week = current_time.weekday()
        weekly_factor = 1.0 if day_of_week < 5 else 0.95
        
        # Random events (occasional spikes)
        event_chance = random.random()
        if event_chance > 0.97:  # 3% chance of anomaly
            spike = random.uniform(50, 150)
            is_anomaly = True
            anomaly_type = random.choice(["rapid_rise", "color_change", "flow_spike"])
        elif event_chance > 0.92:  # 5% chance of minor event
            spike = random.uniform(20, 50)
            is_anomaly = True
            anomaly_type = "rapid_rise"
        else:
            spike = 0
            is_anomaly = False
            anomaly_type = None
        
        # Calculate water level
        noise = random.gauss(0, 10)
        water_level = base_level + base_trend * (current_time - start_time).days
        water_level = water_level * daily_factor * weekly_factor + spike + noise
        water_level = max(50, min(800, water_level))
        
        # Flow class based on water level and randomness
        if water_level > 500 or spike > 80:
            flow_class = "turbulent"
        elif water_level > 400 or spike > 40:
            flow_class = "high"
        elif water_level > 250:
            flow_class = "moderate"
        elif water_level > 150:
            flow_class = "low"
        else:
            flow_class = "still"
        
        # Sediment based on flow
        if flow_class == "turbulent":
            sediment = random.choice(["muddy", "muddy", "silt"])
        elif flow_class == "high":
            sediment = random.choice(["silt", "muddy", "silt"])
        else:
            sediment = random.choice(["clear", "silt", "clear", "green"])
        
        # Turbulence score
        turbulence = {
            "still": random.randint(0, 10),
            "low": random.randint(10, 25),
            "moderate": random.randint(25, 45),
            "high": random.randint(45, 70),
            "turbulent": random.randint(70, 100)
        }.get(flow_class, 30)
        
        # Gauge health (degrades slowly over time)
        days_elapsed = (current_time - start_time).days
        visibility = max(60, 100 - days_elapsed * 0.5 + random.randint(-5, 5))
        algae = random.random() < (days_elapsed / 100)
        
        # Erosion (rare but persistent once detected)
        erosion = random.random() < 0.1
        
        # Overall risk
        if flow_class == "turbulent" or is_anomaly:
            risk = "high"
        elif flow_class == "high" or water_level > 400:
            risk = "medium"
        else:
            risk = "low"
        
        # Create analysis record
        analysis = RiverAnalysis(
            site_id=site_id,
            timestamp=current_time,
            water_color_rgb=json.dumps([
                random.randint(80, 150),
                random.randint(100, 160),
                random.randint(60, 120)
            ]),
            sediment_type=sediment,
            pollution_index=random.uniform(0, 0.4) if sediment != "polluted" else random.uniform(0.6, 1.0),
            flow_speed_class=flow_class,
            turbulence_score=turbulence,
            gauge_visibility_score=int(visibility),
            gauge_damage_detected=visibility < 70,
            damage_type="algae" if algae else None,
            anomaly_detected=is_anomaly,
            anomaly_type=anomaly_type if is_anomaly else None,
            anomaly_description=f"Water level changed by {spike:.0f}cm" if is_anomaly else None,
            erosion_detected=erosion,
            erosion_change_pct=random.uniform(1, 10) if erosion else 0,
            overall_risk=risk,
            ai_analysis_json=json.dumps({
                "water_level": {"estimated_level_cm": round(water_level, 1)},
                "flow_analysis": {"flow_class": flow_class, "turbulence_score": turbulence},
                "water_color": {"sediment_type": sediment},
                "anomalies": {"anomaly_detected": is_anomaly, "anomaly_type": anomaly_type},
                "summary": f"Water level at {water_level:.0f}cm, {flow_class} flow, {sediment} water"
            })
        )
        
        db_session.add(analysis)
        created_count += 1
        
        # Advance time
        hours_to_add = 24 / readings_per_day + random.uniform(-1, 1)
        current_time += timedelta(hours=hours_to_add)
        
        # Update base for trend
        base_level += base_trend * 0.1
        
        # Occasional trend reversal
        if random.random() > 0.95:
            base_trend = -base_trend
    
    db_session.commit()
    return created_count


def generate_mock_alerts(db_session, site_id: int, num_alerts: int = 5) -> int:
    """Generate mock alerts for a site"""
    from models import FloodAlert
    
    alert_types = [
        ("water_level_high", "Water level exceeded danger threshold", "high"),
        ("rapid_rise", "Rapid water level increase detected", "high"),
        ("turbulent_flow", "Turbulent water flow detected", "medium"),
        ("gauge_maintenance", "Gauge requires maintenance", "low"),
        ("color_anomaly", "Unusual water color detected", "medium"),
        ("erosion_warning", "Riverbank erosion signs detected", "medium"),
    ]
    
    created_count = 0
    now = datetime.utcnow()
    
    for i in range(num_alerts):
        alert_type, message, severity = random.choice(alert_types)
        
        alert = FloodAlert(
            site_id=site_id,
            alert_type=alert_type,
            message=message,
            severity=severity,
            created_at=now - timedelta(hours=random.randint(1, 72))
        )
        db_session.add(alert)
        created_count += 1
    
    db_session.commit()
    return created_count


def main():
    """Main function to generate all mock data"""
    print("=" * 60)
    print("River Memory AI - Mock Data Generator")
    print("=" * 60)
    
    # Import Flask app
    from app import create_app
    from extensions import db
    
    app = create_app()
    
    with app.app_context():
        # Generate sites
        print("\n[1/3] Generating mock sites...")
        sites = generate_mock_sites(db.session, num_sites=5)
        print(f"    Created/found {len(sites)} sites")
        
        # Generate analyses for each site
        print("\n[2/3] Generating mock analyses...")
        total_analyses = 0
        for site in sites:
            count = generate_mock_analyses(
                db.session, 
                site["id"], 
                days=30,
                readings_per_day=4
            )
            total_analyses += count
            print(f"    Site {site['id']} ({site['name']}): {count} analyses")
        print(f"    Total: {total_analyses} analyses")
        
        # Generate alerts
        print("\n[3/3] Generating mock alerts...")
        total_alerts = 0
        for site in sites:
            count = generate_mock_alerts(db.session, site["id"], num_alerts=3)
            total_alerts += count
        print(f"    Total: {total_alerts} alerts")
        
        print("\n" + "=" * 60)
        print("Mock data generation complete!")
        print("=" * 60)
        print(f"\nSummary:")
        print(f"  - Sites: {len(sites)}")
        print(f"  - Analyses: {total_analyses}")
        print(f"  - Alerts: {total_alerts}")
        print(f"\nYou can now test the API endpoints:")
        print(f"  GET /api/v1/sites/<site_id>/timeline")
        print(f"  GET /api/v1/sites/<site_id>/summary")


if __name__ == "__main__":
    main()
