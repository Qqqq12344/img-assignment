from shapely.geometry import Polygon, box

def detect_violations(detections, zones):
    violations = []

    print("\n=== DEBUG: Starting Violation Check ===")

    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        vehicle_box = box(x1, y1, x2, y2)

        print(f"\n[Vehicle {idx}]")
        print(f" - Type: {det['cls_name']}")
        print(f" - Confidence: {det['conf']:.2f}")
        print(f" - BBox: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")

        for z_idx, zone in enumerate(zones):
            zone_type = zone.get("type", "unknown")
            zone_points = zone.get("points", [])

            if len(zone_points) < 3:
                print(f"   > Skipping zone {z_idx}: Less than 3 points")
                continue

            zone_polygon = Polygon(zone_points)

            print(f"   > Checking against zone {z_idx} (type: {zone_type})")
            print(f"     - Zone points: {zone_points[:2]}... (total: {len(zone_points)})")

            if vehicle_box.intersects(zone_polygon):
                print("     --> INTERSECTS → Marked as violation ✅")

                violation = {
                    "vehicle_detection_idx": idx,
                    "violation_type": "zone_violation",
                    "zone_type": zone_type,
                    "vehicle_type": det["cls_name"],
                    "vehicle_confidence": det["conf"],
                    "severity": "critical" if zone_type == "no-parking" else "high",
                    "message": f"Vehicle parked in {zone_type.replace('-', ' ').title()} zone"
                }
                violations.append(violation)
                break  # Only one zone violation per vehicle
            else:
                print("     --> NO INTERSECTION ❌")

    print(f"\n=== DEBUG: Violation check completed. Total violations: {len(violations)} ===\n")
    return violations
