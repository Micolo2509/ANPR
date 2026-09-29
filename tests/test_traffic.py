from datetime import datetime, timezone

import numpy as np

from app.traffic import CameraRuleConfig, TrafficStore, TrafficViolationMonitor, estimate_speed_kmh


def timestamp(second: float) -> float:
    return datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() + second


def test_estimate_speed_requires_calibration():
    assert estimate_speed_kmh((0, 0), (10, 0), 1, 0) is None
    assert round(estimate_speed_kmh((0, 0), (10, 0), 1, 10), 1) == 3.6


def test_red_light_requires_crossing_and_deduplicates(tmp_path):
    store = TrafficStore(tmp_path / "traffic.sqlite3", tmp_path / "evidence")
    config = CameraRuleConfig(
        traffic_light_enabled=True,
        traffic_light_state="RED",
        stop_line=[(50, 0), (50, 100)],
    )
    monitor = TrafficViolationMonitor(store, config, min_track_frames=3, confidence_threshold=0.5, ocr_threshold=0.5)
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    assert monitor.observe(7, (10, 10, 20, 20), timestamp(0), "ABC123AB", 0.9, 0.9, frame) == []
    assert monitor.observe(7, (20, 10, 30, 20), timestamp(1), "ABC123AB", 0.9, 0.9, frame) == []
    records = monitor.observe(7, (55, 10, 65, 20), timestamp(2), "ABC123AB", 0.9, 0.9, frame)
    assert len(records) == 1
    assert records[0]["violation_type"] == "RED_LIGHT"
    assert records[0]["violation_status"] == "PENDING_REVIEW"
    assert records[0]["citation_id"] is not None
    assert monitor.observe(7, (60, 10, 70, 20), timestamp(3), "ABC123AB", 0.9, 0.9, frame) == []
    citation = store.get_citation(records[0]["citation_id"])
    assert citation["citation_status"] == "PENDING_REVIEW"
    assert citation["citation_number"].startswith("VNPR-2026-")


def test_bus_lane_needs_duration_and_excludes_allowed_vehicle(tmp_path):
    store = TrafficStore(tmp_path / "traffic.sqlite3")
    config = CameraRuleConfig(bus_lane_enabled=True, bus_lane_polygon=[(0, 0), (100, 0), (100, 100), (0, 100)], bus_lane_min_duration=2)
    monitor = TrafficViolationMonitor(store, config, min_track_frames=2, confidence_threshold=0.5)
    assert monitor.observe(1, (10, 10, 20, 20), timestamp(0), "ABC123AB", 0.9, 0.9) == []
    assert monitor.observe(1, (10, 10, 20, 20), timestamp(1), "ABC123AB", 0.9, 0.9) == []
    records = monitor.observe(1, (10, 10, 20, 20), timestamp(2.1), "ABC123AB", 0.9, 0.9)
    assert records[0]["violation_type"] == "BUS_LANE"

    allowed = TrafficViolationMonitor(store, config, min_track_frames=2, confidence_threshold=0.5)
    allowed.observe(2, (10, 10, 20, 20), timestamp(0), "XYZ123XY", 0.9, 0.9, vehicle_type="bus")
    allowed.observe(2, (10, 10, 20, 20), timestamp(3), "XYZ123XY", 0.9, 0.9, vehicle_type="bus")
    assert not any(item["track_id"] == 2 for item in store.list_violations())


def test_speeding_requires_calibrated_zone(tmp_path):
    store = TrafficStore(tmp_path / "traffic.sqlite3")
    config = CameraRuleConfig(speed_limit=50, speed_tolerance=5, speed_measurement_zone=[(0, 0), (100, 0), (100, 100), (0, 100)], pixels_per_meter=10)
    monitor = TrafficViolationMonitor(store, config, min_track_frames=2, confidence_threshold=0.5)
    monitor.observe(3, (10, 10, 20, 20), timestamp(0), "KJA123AA", 0.9, 0.9)
    records = monitor.observe(3, (190, 10, 200, 20), timestamp(1), "KJA123AA", 0.9, 0.9)
    assert records == []

    uncalibrated = CameraRuleConfig(speed_limit=50, speed_measurement_zone=[(0, 0), (100, 0), (100, 100), (0, 100)])
    no_speed = TrafficViolationMonitor(store, uncalibrated, min_track_frames=2, confidence_threshold=0.5)
    no_speed.observe(4, (10, 10, 20, 20), timestamp(0), "KJA123AA", 0.9, 0.9)
    assert no_speed.observe(4, (20, 10, 30, 20), timestamp(1), "KJA123AA", 0.9, 0.9) == []


def test_review_changes_violation_and_citation_status(tmp_path):
    store = TrafficStore(tmp_path / "traffic.sqlite3")
    config = CameraRuleConfig(traffic_light_enabled=True, traffic_light_state="RED", stop_line=[(50, 0), (50, 100)])
    monitor = TrafficViolationMonitor(store, config, min_track_frames=2, confidence_threshold=0.5)
    monitor.observe(5, (10, 10, 20, 20), timestamp(0), "ABC123AB", 0.9, 0.9)
    records = monitor.observe(5, (60, 10, 70, 20), timestamp(1), "ABC123AB", 0.9, 0.9)
    reviewed = store.review(records[0]["id"], "confirm", 42)
    assert reviewed["violation_status"] == "CONFIRMED"
    assert store.get_citation(records[0]["citation_id"])["citation_status"] == "APPROVED"


def test_citation_edit_and_statistics(tmp_path):
    store = TrafficStore(tmp_path / "traffic.sqlite3")
    config = CameraRuleConfig(traffic_light_enabled=True, traffic_light_state="RED", stop_line=[(50, 0), (50, 100)], citation_amounts={"RED_LIGHT": 25000})
    monitor = TrafficViolationMonitor(store, config, min_track_frames=2, confidence_threshold=0.5)
    monitor.observe(9, (10, 10, 20, 20), timestamp(0), "ABC123AB", 0.9, 0.9)
    records = monitor.observe(9, (60, 10, 70, 20), timestamp(1), "ABC123AB", 0.9, 0.9)
    citation_id = records[0]["citation_id"]
    edited = store.update_citation(citation_id, 42, amount=25000, due_date="2026-12-31", description="Reviewed application amount")
    assert edited["amount"] == 25000
    assert edited["citation_status"] == "PENDING_REVIEW"
    assert store.statistics()["by_type"]["RED_LIGHT"] == 1
    assert store.statistics()["pending_reviews"] == 1
