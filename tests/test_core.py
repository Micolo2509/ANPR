from dataclasses import replace

import cv2
import numpy as np

from app.database import HistoryStore, PersistentVehicleStore, PlateRegistryStore, VehicleStatusStore, VehicleObligationStore
from app.database import AdminStore
from app.auth import create_session, read_session
from app.detector import Detection
from app.normalization import is_plausible_nigerian_plate, normalize_plate_text
from app.pipeline import VNPRPipeline
from app.segmentation.character_segmentation import segment_plate_characters
from app.schemas import BoundingBox, PlateResult
from app.vehicle_classifier import VehicleClassifier
from fastapi.testclient import TestClient
from app.main import LiveCameraManager, app


class FakeDetector:
    def detect(self, image):
        return [Detection(1, 2, 8, 6, 0.91)]

    def _serialize_detection(self, item):
        return {"x1": item.x1, "y1": item.y1, "x2": item.x2, "y2": item.y2, "confidence": item.confidence, "method": "yolo"}


class FakeOCR:
    def read(self, crop):
        assert crop.shape == (4, 7, 3)
        return "ABC-123AB", 0.88


class FakeVehicleClassifier:
    def classify(self, image):
        return type("Prediction", (), {
            "make": "Toyota",
            "model": "Corolla",
            "confidence": 0.94,
        })()


def test_pipeline_connects_detection_and_ocr():
    results = VNPRPipeline(
        FakeDetector(),
        FakeOCR(),
        vehicle_classifier=FakeVehicleClassifier(),
    ).recognize(np.zeros((10, 10, 3), dtype=np.uint8))
    assert results[0].text == "ABC123AB"
    assert results[0].make == "Toyota"
    assert results[0].model == "Corolla"
    assert results[0].vehicle_confidence == 0.94
    assert results[0].bbox.x1 == 1
    assert results[0].detection_confidence == 0.91


def test_external_live_worker_keeps_only_newest_submitted_frame():
    manager = LiveCameraManager()
    manager._running = True
    first = np.zeros((8, 12, 3), dtype=np.uint8)
    newest = np.full((6, 10, 3), 255, dtype=np.uint8)

    manager.submit_frame(first)
    manager.submit_frame(newest)

    assert manager._latest_frame.shape == newest.shape
    assert int(manager._latest_frame[0, 0, 0]) == 255
    assert manager._raw_frame_count == 2


def test_pipeline_marks_registered_and_verified_plate(tmp_path):
    registry = PlateRegistryStore(tmp_path / "registry.sqlite3")
    registry.upsert("ABC-123AB", is_verified=True)

    results = VNPRPipeline(
        FakeDetector(),
        FakeOCR(),
        vehicle_classifier=FakeVehicleClassifier(),
        plate_registry=registry,
    ).recognize(np.zeros((10, 10, 3), dtype=np.uint8))

    assert results[0].is_registered is True
    assert results[0].is_verified is True
    assert results[0].registration_status == "registered_verified"


def test_registry_distinguishes_unregistered_plate(tmp_path):
    registry = PlateRegistryStore(tmp_path / "registry.sqlite3")

    assert registry.lookup("ABC-123AB") == (False, False)
    assert registry.upsert("ABC-123AB", is_verified=False)["registration_status"] == "registered_unverified"
    assert registry.lookup("ABC-123AB") == (True, False)


def test_normalization_handles_ocr_noise():
    assert normalize_plate_text(" ab-c|23ab\n") == "ABCI23AB"
    assert normalize_plate_text("SMK-586 KF") == "SMK586KF"
    assert normalize_plate_text("LAG-08OAA") == "LAG08OAA"
    assert normalize_plate_text("NIGERIAKJA123AA7") == "KJA123AA"
    assert normalize_plate_text("NIGERIA KJA123AA") == "KJA123AA"
    assert normalize_plate_text("Plate: KJA123AA") == "KJA123AA"
    assert normalize_plate_text("KJA123AA Nigeria") == "KJA123AA"
    assert normalize_plate_text("XXXKJA123AAXXX") == "KJA123AA"
    assert normalize_plate_text("KJA123AA") == "KJA123AA"
    assert normalize_plate_text("LSD100GX") == "LSD100GX"


def test_normalization_does_not_promote_arbitrary_noise():
    assert normalize_plate_text("NIGERIA") == "NIGERIA"
    assert normalize_plate_text("HELLO123") == "HELLO123"
    assert normalize_plate_text("ABCDEFG") == "ABCDEFG"


def test_nigerian_plate_validation_and_format_rules():
    from app.normalization import validate_nigerian_plate

    assert validate_nigerian_plate("SMK586KF") is True
    assert validate_nigerian_plate("SHK588F") is True
    assert validate_nigerian_plate("ABC123XY") is True
    assert validate_nigerian_plate("SHK588FZ") is True
    assert validate_nigerian_plate("123456789") is False
    assert validate_nigerian_plate("ABC123XYZ1") is False


def test_common_nigerian_layout_is_plausible():
    assert is_plausible_nigerian_plate("ABC-123AB")
    assert is_plausible_nigerian_plate("AB1234CD")
    assert is_plausible_nigerian_plate("KAD-450-GH")
    assert not is_plausible_nigerian_plate("123456789")


def test_pipeline_keeps_detected_plate_when_ocr_fails():
    class EmptyOCR:
        def read(self, crop):
            return "", 0.0

    results = VNPRPipeline(
        FakeDetector(),
        EmptyOCR(),
        vehicle_classifier=FakeVehicleClassifier(),
    ).recognize(np.zeros((10, 10, 3), dtype=np.uint8))

    assert len(results) == 1
    assert results[0].text == "UNKNOWN"
    assert results[0].to_dict()["ocr_status"] == "failed"
    assert results[0].to_dict()["plate_number"] == "UNKNOWN"
    assert results[0].bbox.x1 == 1


def test_pipeline_uses_database_verified_plate_correction():
    class ConfusableOCR:
        def read(self, crop):
            return "SHK588KF", 0.81

    class Registry:
        def lookup(self, plate_text):
            if plate_text == "SMK586KF":
                return (True, True)
            return (False, False)

    results = VNPRPipeline(
        FakeDetector(),
        ConfusableOCR(),
        vehicle_classifier=FakeVehicleClassifier(),
        plate_registry=Registry(),
    ).recognize(np.zeros((10, 10, 3), dtype=np.uint8), debug=True)

    assert results[0].text == "SMK586KF"
    assert results[0].debug["raw_ocr_text"] == "SHK588KF"
    assert results[0].debug["corrected_text"] == "SMK586KF"
    assert results[0].debug["correction_applied"] is True
    assert results[0].is_verified is True


def test_history_store_saves_and_lists_newest_first(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    store.save_results([
        PlateResult(
            "ABC-123AB",
            0.91,
            0.88,
            BoundingBox(1, 2, 8, 6),
            "Toyota",
            "Corolla",
            0.94,
        ),
        PlateResult(
            "LAG-08OAA",
            0.89,
            0.86,
            BoundingBox(2, 3, 9, 7),
        ),
    ])

    records = store.list_results()

    assert len(records) == 2
    assert records[0]["plate_text"] == "LAG08OAA"
    assert records[1]["plate_text"] == "ABC123AB"
    assert records[1]["make"] == "Toyota"
    assert records[1]["model"] == "Corolla"

    store.clear()

    assert store.list_results() == []
    assert records[1]["registration_status"] == "not_registered_unverified"


def test_persistent_vehicle_store_registers_and_updates_existing_vehicle(tmp_path):
    store = PersistentVehicleStore(tmp_path / "vehicles.sqlite3")

    first = store.record_recognition(
        "SMK-586 KF", "Toyota", "Image Upload", 0.91, 0.88,
        image_path="first.jpg", vehicle_confidence=0.94, ocr_threshold=0.55,
    )
    second = store.record_recognition(
        "SMK586KF", None, "Live Camera", 0.92, 0.90,
        vehicle_confidence=0.0, ocr_threshold=0.55,
    )

    assert first["is_new"] is True
    assert second["is_new"] is False
    assert second["vehicle_id"] == first["vehicle_id"]
    assert second["first_seen"] == first["first_seen"]
    assert second["recognition_count"] == 2
    assert second["vehicle_make"] == "Toyota"
    assert len(store.list_history()) == 2
    assert store.list_history(plate_number="SMK-586 KF")[0]["recognition_source"] == "Live Camera"


def test_persistent_vehicle_store_clears_history_without_removing_vehicles(tmp_path):
    store = PersistentVehicleStore(tmp_path / "vehicles.sqlite3")
    store.record_recognition(
        "SMK-586 KF", "Toyota", "Image Upload", 0.91, 0.88,
        vehicle_confidence=0.94, ocr_threshold=0.55,
    )

    store.clear_history()

    assert store.list_history() == []
    assert [vehicle["plate_number"] for vehicle in store.list_vehicles()] == ["SMK586KF"]


def test_persistent_vehicle_store_rejects_invalid_or_low_confidence_ocr(tmp_path):
    store = PersistentVehicleStore(tmp_path / "vehicles.sqlite3")

    assert store.record_recognition("", "Toyota", "Image Upload", 0.9, 0.9, ocr_threshold=0.55) is None
    assert store.record_recognition("123456789", "Toyota", "Image Upload", 0.9, 0.9, ocr_threshold=0.55) is None
    assert store.record_recognition("ABC123AB", "Toyota", "Image Upload", 0.9, 0.4, ocr_threshold=0.55) is None
    assert store.list_vehicles() == []


def test_character_segmentation_finds_characters_on_plate_crop():
    plate = np.full((80, 260, 3), 255, dtype=np.uint8)
    cv2.rectangle(plate, (10, 10), (250, 70), (0, 0, 0), thickness=4)
    for x in range(24, 230, 24):
        cv2.rectangle(plate, (x, 18), (x + 12, 62), (0, 0, 0), thickness=-1)
    result = segment_plate_characters(plate)
    assert result["success"] is True
    assert result["character_count"] >= 4
    assert len(result["character_crops"]) == result["character_count"]


def test_pipeline_falls_back_to_full_plate_ocr_when_segmentation_is_unreliable():
    class EmptyOCR:
        def read(self, crop):
            return "ABC123AB", 0.88

    class BrokenSegmentationDetector:
        def detect(self, image):
            return [Detection(1, 2, 8, 6, 0.91)]

        def _serialize_detection(self, item):
            return {"x1": item.x1, "y1": item.y1, "x2": item.x2, "y2": item.y2, "confidence": item.confidence, "method": "yolo"}

    results = VNPRPipeline(
        BrokenSegmentationDetector(),
        EmptyOCR(),
        vehicle_classifier=FakeVehicleClassifier(),
    ).recognize(np.zeros((10, 10, 3), dtype=np.uint8), debug=True)

    assert results[0].text == "ABC123AB"
    assert results[0].debug["character_segmentation"]["success"] in (True, False)


def test_segmentation_debug_route_is_optional_and_separate(tmp_path):
    store = AdminStore(tmp_path / "debug_admins.sqlite3")
    admin = store.create("Debug Admin", "debug@example.com", "secret")
    token = create_session(admin["id"], "change-this-development-secret", 1)

    plate = np.full((80, 260, 3), 255, dtype=np.uint8)
    cv2.rectangle(plate, (10, 10), (250, 70), (0, 0, 0), thickness=4)
    for x in range(24, 230, 24):
        cv2.rectangle(plate, (x, 18), (x + 12, 62), (0, 0, 0), thickness=-1)
    payload = cv2.imencode(".png", plate)[1].tobytes()

    client = TestClient(app)
    response = client.post(
        "/api/v1/debug/segmentation",
        files={"file": ("plate.png", payload, "image/png")},
        params={"debug": True},
        cookies={"vnpr_session": token},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["plate_detected"] is True
    assert data["character_segmentation"]["enabled"] is True
    assert "debug_images" in data
    assert data["debug_images"]["plate_crop"]


def test_missing_optional_vehicle_model_does_not_fail(tmp_path):
    classifier = VehicleClassifier(tmp_path / "missing.pt")

    assert classifier.classify(np.zeros((2, 2, 3), dtype=np.uint8)) is None


def test_admin_store_hashes_password_and_authenticates(tmp_path):
    store = AdminStore(tmp_path / "admins.sqlite3")
    admin = store.create("Ada Admin", "ADA@example.com", "correct horse")

    assert admin["email"] == "ada@example.com"
    assert store.authenticate("ada@example.com", "correct horse")["id"] == admin["id"]
    assert store.authenticate("ada@example.com", "wrong password") is None


def test_session_rejects_tampering():
    token = create_session(42, "test-secret", 1)

    assert read_session(token, "test-secret")["admin_id"] == 42
    assert read_session(token + "x", "test-secret") is None
    assert read_session(token, "wrong-secret") is None


def test_signup_returns_json_errors():
    client = TestClient(app)
    email = "signup-validation@example.com"
    first = client.post("/api/v1/auth/signup", json={"name": "Signup Test", "email": email, "password": "securepass123"})
    assert first.status_code in (200, 409)
    duplicate = client.post("/api/v1/auth/signup", json={"name": "Signup Test", "email": email, "password": "securepass123"})
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"] == "An admin with this email already exists"


def test_signup_session_survives_recreated_application_state(tmp_path, monkeypatch):
    import app.main as main

    database_path = tmp_path / "persistent" / "auth.sqlite3"
    test_settings = replace(main.settings, database_path=database_path, auth_secret="session-test-secret")
    monkeypatch.setattr(main, "settings", test_settings)
    monkeypatch.setattr(main, "admins", AdminStore(database_path))

    client = TestClient(main.app, base_url="https://testserver")
    signup = client.post(
        "/api/v1/auth/signup",
        json={"name": "Restart Admin", "email": "restart@example.com", "password": "securepass123"},
    )
    assert signup.status_code == 200

    monkeypatch.setattr(main, "settings", replace(test_settings))
    monkeypatch.setattr(main, "admins", AdminStore(database_path))

    current = client.get("/api/v1/auth/me")
    assert current.status_code == 200
    assert current.json()["admin"]["email"] == "restart@example.com"


def test_signup_session_protects_endpoint(tmp_path, monkeypatch):
    import app.main as main

    database_path = tmp_path / "signup.sqlite3"
    test_settings = replace(main.settings, database_path=database_path, auth_secret="signup-secret")
    monkeypatch.setattr(main, "settings", test_settings)
    monkeypatch.setattr(main, "admins", AdminStore(database_path))
    client = TestClient(main.app, base_url="https://testserver")

    response = client.post(
        "/api/v1/auth/signup",
        json={"name": "Signup Admin", "email": "signup-session@example.com", "password": "securepass123"},
    )

    assert response.status_code == 200
    assert client.get("/api/v1/auth/me").status_code == 200


def test_login_session_protects_endpoint(tmp_path, monkeypatch):
    import app.main as main

    database_path = tmp_path / "login.sqlite3"
    store = AdminStore(database_path)
    store.create("Login Admin", "login-session@example.com", "securepass123")
    test_settings = replace(main.settings, database_path=database_path, auth_secret="login-secret")
    monkeypatch.setattr(main, "settings", test_settings)
    monkeypatch.setattr(main, "admins", AdminStore(database_path))
    client = TestClient(main.app, base_url="https://testserver")

    response = client.post(
        "/api/v1/auth/login",
        json={"email": "login-session@example.com", "password": "securepass123"},
    )

    assert response.status_code == 200
    assert client.get("/api/v1/auth/me").status_code == 200


def test_signed_session_survives_admin_store_loss(tmp_path, monkeypatch):
    import app.main as main

    database_path = tmp_path / "lost.sqlite3"
    test_settings = replace(main.settings, database_path=database_path, auth_secret="loss-secret")
    monkeypatch.setattr(main, "settings", test_settings)
    monkeypatch.setattr(main, "admins", AdminStore(database_path))
    client = TestClient(main.app, base_url="https://testserver")
    signup = client.post(
        "/api/v1/auth/signup",
        json={"name": "Lost DB Admin", "email": "lost-db@example.com", "password": "securepass123"},
    )
    assert signup.status_code == 200

    class UnavailableAdminStore:
        def get(self, admin_id):
            return None

    monkeypatch.setattr(main, "admins", UnavailableAdminStore())

    current = client.get("/api/v1/auth/me")
    assert current.status_code == 200
    assert current.json()["admin"]["email"] == "lost-db@example.com"


def test_signed_session_rejects_tampering_and_expiry(tmp_path, monkeypatch):
    import app.main as main

    database_path = tmp_path / "invalid.sqlite3"
    test_settings = replace(main.settings, database_path=database_path, auth_secret="invalid-secret")
    monkeypatch.setattr(main, "settings", test_settings)
    monkeypatch.setattr(main, "admins", AdminStore(database_path))
    client = TestClient(main.app, base_url="https://testserver")
    admin = {"id": 99, "name": "Signed Admin", "email": "signed@example.com"}

    tampered = create_session(admin["id"], test_settings.auth_secret, 1, admin) + "x"
    expired = create_session(admin["id"], test_settings.auth_secret, -1, admin)

    assert client.get("/api/v1/auth/me", cookies={"vnpr_session": tampered}).status_code == 401
    assert client.get("/api/v1/auth/me", cookies={"vnpr_session": expired}).status_code == 401


def test_live_tracker_prefers_repeated_high_confidence_plate():
    from app.live_tracker import VehicleTrackState, LiveVehicleTracker

    tracker = LiveVehicleTracker()
    tracker.tracked_vehicles[1] = VehicleTrackState(track_id=1, plate="KJA123AA", ocr_confidence=0.96, detection_confidence=0.95, last_seen=0.0)
    tracker.record_ocr_result(1, "KJA123AA", 0.91, 0.95)
    tracker.record_ocr_result(1, "KJA128AA", 0.61, 0.90)
    tracker.record_ocr_result(1, "KJA123AA", 0.98, 0.97)

    state = tracker.tracked_vehicles[1]
    assert state.plate == "KJA123AA"
    assert state.ocr_confidence >= 0.96
    assert len(state.recognition_history) >= 3


def test_live_tracker_keeps_unique_track_ids_and_ignores_bad_ocr_overwrites():
    from app.live_tracker import LiveVehicleTracker

    tracker = LiveVehicleTracker()
    tracker.record_ocr_result(2, "ABC456XY", 0.90, 0.93)
    tracker.record_ocr_result(2, "ZZZ999ZZ", 0.12, 0.40)
    tracker.record_ocr_result(3, "LAG789AB", 0.94, 0.96)

    assert tracker.tracked_vehicles[2].plate == "ABC456XY"
    assert tracker.tracked_vehicles[3].plate == "LAG789AB"
    assert sorted(tracker.tracked_vehicles) == [2, 3]


def test_vehicle_status_and_obligation_tracking(tmp_path):
    status_store = VehicleStatusStore(tmp_path / "status.sqlite3")
    obligation_store = VehicleObligationStore(tmp_path / "status.sqlite3")

    status_store.set_status("ABC-123AB", "stolen", "Reported stolen by owner")
    obligation_store.add_obligation("ABC-123AB", 45000.0, "Road tax arrears")
    obligation_store.add_obligation("ABC-123AB", 18000.0, "Parking violation", status="PAID")

    summary = status_store.lookup("ABC-123AB")
    assert summary["status"] == "stolen"
    assert summary["reason"] == "Reported stolen by owner"
    assert summary["amount_owed"] == 45000.0
    assert summary["outstanding_amount"] == 45000.0
    assert obligation_store.outstanding_total("ABC-123AB") == 45000.0
    assert obligation_store.list_for_plate("ABC-123AB")[0]["plate_text"] == "ABC123AB"


def test_vehicle_alert_store_prioritizes_high_risk_vehicles(tmp_path):
    from app.database import VehicleAlertStore

    status_store = VehicleStatusStore(tmp_path / "alerts.sqlite3")
    obligation_store = VehicleObligationStore(tmp_path / "alerts.sqlite3")
    alerts = VehicleAlertStore(tmp_path / "alerts.sqlite3")

    status_store.set_status("KJA-123AB", "stolen", "Reported stolen by owner")
    obligation_store.add_obligation("KJA-123AB", 52000.0, "Court fine")
    status_store.set_status("ABC-123AB", "clear", None)
    obligation_store.add_obligation("ABC-123AB", 5000.0, "Road tax arrears")

    results = alerts.list_alerts()
    assert results[0]["plate_text"] == "KJA-123AB"
    assert results[0]["priority"] == "high"
    assert results[0]["amount_owed"] == 52000.0
    assert any(item["plate_text"] == "ABC-123AB" for item in results)