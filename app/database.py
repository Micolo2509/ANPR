import sqlite3
import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .normalization import format_plate_text, is_plausible_nigerian_plate, normalize_plate_text
from .schemas import PlateResult


class HistoryStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS recognition_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recognized_at TEXT NOT NULL,
                    plate_text TEXT NOT NULL,
                    detection_confidence REAL NOT NULL,
                    ocr_confidence REAL NOT NULL,
                    make TEXT,
                    model TEXT,
                    vehicle_confidence REAL,
                    x1 INTEGER NOT NULL,
                    y1 INTEGER NOT NULL,
                    x2 INTEGER NOT NULL,
                    y2 INTEGER NOT NULL
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(recognition_history)")
            }
            if "is_registered" not in columns:
                connection.execute(
                    "ALTER TABLE recognition_history ADD COLUMN is_registered INTEGER NOT NULL DEFAULT 0"
                )
            if "is_verified" not in columns:
                connection.execute(
                    "ALTER TABLE recognition_history ADD COLUMN is_verified INTEGER NOT NULL DEFAULT 0"
                )

    def save_results(self, results: list[PlateResult]) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        rows = [
            (
                timestamp,
                normalize_plate_text(result.text),
                result.detection_confidence,
                result.ocr_confidence,
                result.make,
                result.model,
                result.vehicle_confidence,
                int(result.is_registered),
                int(result.is_verified),
                result.bbox.x1,
                result.bbox.y1,
                result.bbox.x2,
                result.bbox.y2,
            )
            for result in results
            if normalize_plate_text(result.text)
        ]
        if not rows:
            return
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO recognition_history (
                    recognized_at, plate_text, detection_confidence,
                    ocr_confidence, make, model, vehicle_confidence,
                    is_registered, is_verified, x1, y1, x2, y2
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def list_results(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, recognized_at, plate_text, detection_confidence,
                       ocr_confidence, make, model, vehicle_confidence,
                       is_registered, is_verified,
                       x1, y1, x2, y2
                FROM recognition_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        records = []
        for row in rows:
            record = dict(row)
            record["plate_text"] = normalize_plate_text(record["plate_text"])
            record["is_registered"] = bool(record["is_registered"])
            record["is_verified"] = bool(record["is_verified"])
            record["registration_status"] = (
                "registered_verified"
                if record["is_registered"] and record["is_verified"]
                else "registered_unverified"
                if record["is_registered"]
                else "not_registered_unverified"
            )
            records.append(record)
        return records

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM recognition_history")


class PersistentVehicleStore:
    """Stores one durable vehicle profile and one row for each successful scan."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS vehicles (
                    vehicle_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT NOT NULL UNIQUE,
                    vehicle_make TEXT,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    recognition_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'Registered'
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS vehicle_recognition_history (
                    recognition_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vehicle_id INTEGER NOT NULL,
                    plate_number TEXT NOT NULL,
                    vehicle_make TEXT,
                    recognition_source TEXT NOT NULL,
                    recognized_at TEXT NOT NULL,
                    detection_confidence REAL NOT NULL,
                    ocr_confidence REAL NOT NULL,
                    image_path TEXT,
                    FOREIGN KEY (vehicle_id) REFERENCES vehicles(vehicle_id)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_vehicle_history_plate ON vehicle_recognition_history(plate_number)"
            )

    @staticmethod
    def _make_is_reliable(make: str | None, vehicle_confidence: float | None) -> bool:
        value = str(make or "").strip().lower()
        return bool(value and value not in {"unknown", "unidentified", "none"} and float(vehicle_confidence or 0.0) >= 0.55)

    def record_recognition(
        self,
        plate_number: str,
        vehicle_make: str | None,
        recognition_source: str,
        detection_confidence: float,
        ocr_confidence: float,
        image_path: str | None = None,
        vehicle_confidence: float | None = None,
        ocr_threshold: float = 0.0,
    ) -> dict[str, Any] | None:
        normalized_plate = normalize_plate_text(plate_number)
        if (
            not normalized_plate
            or not is_plausible_nigerian_plate(normalized_plate)
            or float(ocr_confidence) < float(ocr_threshold)
            or recognition_source not in {"Image Upload", "Live Camera"}
        ):
            return None

        now = datetime.now(timezone.utc).isoformat()
        reliable_make = self._make_is_reliable(vehicle_make, vehicle_confidence)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO vehicles (plate_number, vehicle_make, first_seen, last_seen, recognition_count, status)
                VALUES (?, ?, ?, ?, 0, 'Registered')
                ON CONFLICT(plate_number) DO NOTHING
                """,
                (normalized_plate, vehicle_make if reliable_make else None, now, now),
            )
            vehicle = connection.execute(
                "SELECT * FROM vehicles WHERE plate_number = ?", (normalized_plate,)
            ).fetchone()
            if vehicle is None:
                raise RuntimeError("Vehicle record could not be created")
            make = vehicle["vehicle_make"]
            if (not make or str(make).strip().lower() in {"unknown", "unidentified"}) and reliable_make:
                make = str(vehicle_make).strip()
                connection.execute(
                    "UPDATE vehicles SET vehicle_make = ? WHERE vehicle_id = ?",
                    (make, vehicle["vehicle_id"]),
                )
            connection.execute(
                """
                UPDATE vehicles
                SET last_seen = ?, recognition_count = recognition_count + 1, status = 'Registered'
                WHERE vehicle_id = ?
                """,
                (now, vehicle["vehicle_id"]),
            )
            updated = connection.execute(
                "SELECT * FROM vehicles WHERE vehicle_id = ?", (vehicle["vehicle_id"],)
            ).fetchone()
            connection.execute(
                """
                INSERT INTO vehicle_recognition_history (
                    vehicle_id, plate_number, vehicle_make, recognition_source,
                    recognized_at, detection_confidence, ocr_confidence, image_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    updated["vehicle_id"], normalized_plate, updated["vehicle_make"], recognition_source,
                    now, float(detection_confidence), float(ocr_confidence), image_path,
                ),
            )
        payload = dict(updated)
        payload.update({"is_new": int(vehicle["recognition_count"]) == 0, "recognized_at": now})
        return payload

    def list_vehicles(self, limit: int = 500) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM vehicles ORDER BY last_seen DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def get_vehicle(self, plate_number: str) -> dict[str, Any] | None:
        normalized_plate = normalize_plate_text(plate_number)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM vehicles WHERE plate_number = ?", (normalized_plate,)
            ).fetchone()
        return dict(row) if row else None

    def list_history(self, limit: int = 500, plate_number: str | None = None) -> list[dict[str, Any]]:
        normalized_plate = normalize_plate_text(plate_number or "")
        query = """
            SELECT recognition_id, vehicle_id, plate_number, vehicle_make,
                   recognition_source, recognized_at, detection_confidence,
                   ocr_confidence, image_path
            FROM vehicle_recognition_history
        """
        if normalized_plate:
            query += " WHERE plate_number = ? ORDER BY recognition_id DESC LIMIT ?"
            parameters: tuple[Any, ...] = (normalized_plate, limit)
        else:
            query += " ORDER BY recognition_id DESC LIMIT ?"
            parameters = (limit,)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def clear_history(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM vehicle_recognition_history")


class PlateRegistryStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS registered_plates (
                    plate_text TEXT PRIMARY KEY,
                    is_verified INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def lookup(self, plate_text: str) -> tuple[bool, bool]:
        normalized_plate = normalize_plate_text(plate_text)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT is_verified FROM registered_plates WHERE plate_text = ?",
                (normalized_plate,),
            ).fetchone()
        return (row is not None, bool(row["is_verified"]) if row else False)

    def upsert(self, plate_text: str, is_verified: bool) -> dict[str, Any]:
        normalized_plate = normalize_plate_text(plate_text)
        updated_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO registered_plates (plate_text, is_verified, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(plate_text) DO UPDATE SET
                    is_verified = excluded.is_verified,
                    updated_at = excluded.updated_at
                """,
                (normalized_plate, int(is_verified), updated_at),
            )
        return {
            "plate_text": normalized_plate,
            "is_registered": True,
            "is_verified": is_verified,
            "registration_status": "registered_verified" if is_verified else "registered_unverified",
            "updated_at": updated_at,
        }

    def remove(self, plate_text: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM registered_plates WHERE plate_text = ?", (plate_text,)
            )
        return cursor.rowcount > 0

    def list_all(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT plate_text, is_verified, updated_at FROM registered_plates ORDER BY plate_text"
            ).fetchall()
        return [
            {
                "plate_text": normalize_plate_text(row["plate_text"]),
                "is_registered": True,
                "is_verified": bool(row["is_verified"]),
                "registration_status": "registered_verified" if row["is_verified"] else "registered_unverified",
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]


class VehicleStatusStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def normalize_status(value: str | None) -> str:
        cleaned = str(value or "clear").strip().lower()
        mapping = {
            "clear": "clear",
            "ok": "clear",
            "registered": "clear",
            "safe": "clear",
            "stolen": "stolen",
            "hot": "stolen",
            "flagged": "flagged",
            "warning": "flagged",
            "alert": "flagged",
            "unregistered": "unregistered",
            "not_registered": "unregistered",
            "not-registered": "unregistered",
        }
        return mapping.get(cleaned, "clear")

    def _initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS vehicle_status (
                    plate_text TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'clear',
                    reason TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def set_status(self, plate_text: str, status: str, reason: str | None = None) -> dict[str, Any]:
        normalized_plate = normalize_plate_text(plate_text)
        if not normalized_plate:
            raise ValueError("Provide a valid plate number")
        normalized_status = self.normalize_status(status)
        updated_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO vehicle_status (plate_text, status, reason, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(plate_text) DO UPDATE SET
                    status = excluded.status,
                    reason = excluded.reason,
                    updated_at = excluded.updated_at
                """,
                (normalized_plate, normalized_status, (reason or "").strip() or None, updated_at),
            )
        lookup = self.lookup(normalized_plate)
        return lookup

    def lookup(self, plate_text: str) -> dict[str, Any]:
        normalized_plate = normalize_plate_text(plate_text)
        if not normalized_plate:
            return {
                "plate_text": "",
                "status": "clear",
                "reason": None,
                "updated_at": None,
                "amount_owed": 0.0,
                "outstanding_amount": 0.0,
                "is_flagged": False,
            }
        with self._connect() as connection:
            row = connection.execute(
                "SELECT plate_text, status, reason, updated_at FROM vehicle_status WHERE plate_text = ?",
                (normalized_plate,),
            ).fetchone()
        obligations = VehicleObligationStore(self.database_path).outstanding_total(normalized_plate)
        if row is None:
            payload = {
                "plate_text": normalized_plate,
                "status": "clear",
                "reason": None,
                "updated_at": None,
                "amount_owed": obligations,
                "outstanding_amount": obligations,
                "is_flagged": False,
            }
            return payload
        status = self.normalize_status(row["status"])
        payload = {
            "plate_text": normalized_plate,
            "status": status,
            "reason": row["reason"],
            "updated_at": row["updated_at"],
            "amount_owed": obligations,
            "outstanding_amount": obligations,
            "is_flagged": status != "clear",
        }
        return payload


class VehicleObligationStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS vehicle_obligations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_text TEXT NOT NULL,
                    amount REAL NOT NULL DEFAULT 0,
                    reason TEXT,
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    created_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def normalize_status(value: str | None) -> str:
        cleaned = str(value or "open").strip().upper()
        if cleaned in {"PAID", "CANCELLED", "CLEARED"}:
            return cleaned
        return "OPEN"

    def add_obligation(self, plate_text: str, amount: float, reason: str | None = None, status: str = "OPEN") -> dict[str, Any]:
        normalized_plate = normalize_plate_text(plate_text)
        if not normalized_plate:
            raise ValueError("Provide a valid plate number")
        obligation_amount = float(amount or 0.0)
        if obligation_amount < 0:
            raise ValueError("Amount owed cannot be negative")
        created_at = datetime.now(timezone.utc).isoformat()
        normalized_status = self.normalize_status(status)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO vehicle_obligations (plate_text, amount, reason, status, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (normalized_plate, obligation_amount, (reason or "").strip() or None, normalized_status, created_at),
            )
        return {
            "id": cursor.lastrowid,
            "plate_text": normalized_plate,
            "amount": obligation_amount,
            "reason": (reason or "").strip() or None,
            "status": normalized_status,
            "created_at": created_at,
        }

    def list_for_plate(self, plate_text: str) -> list[dict[str, Any]]:
        normalized_plate = normalize_plate_text(plate_text)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, plate_text, amount, reason, status, created_at FROM vehicle_obligations WHERE plate_text = ? ORDER BY created_at DESC",
                (normalized_plate,),
            ).fetchall()
        return [dict(row) for row in rows]

    def outstanding_total(self, plate_text: str) -> float:
        normalized_plate = normalize_plate_text(plate_text)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(SUM(amount), 0) AS total FROM vehicle_obligations WHERE plate_text = ? AND status NOT IN ('PAID', 'CANCELLED')",
                (normalized_plate,),
            ).fetchone()
        return float(row["total"] if row else 0.0)


class VehicleAlertStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS vehicle_alert_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_text TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'clear',
                    reason TEXT,
                    amount_owed REAL NOT NULL DEFAULT 0,
                    priority TEXT NOT NULL DEFAULT 'low',
                    evidence TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _priority_for(status: str | None, amount_owed: float) -> str:
        normalized_status = VehicleStatusStore.normalize_status(status)
        if normalized_status in {"stolen", "flagged"} or float(amount_owed or 0.0) >= 50000.0:
            return "high"
        if normalized_status == "unregistered" or float(amount_owed or 0.0) > 0.0:
            return "medium"
        return "low"

    def _build_alert(self, plate_text: str, status: str | None, reason: str | None, amount_owed: float, evidence: str | None = None) -> dict[str, Any]:
        normalized_plate = normalize_plate_text(plate_text)
        if not normalized_plate:
            return {}
        priority = self._priority_for(status, amount_owed)
        return {
            "plate_text": format_plate_text(normalized_plate),
            "status": VehicleStatusStore.normalize_status(status),
            "reason": (reason or "").strip() or None,
            "amount_owed": float(amount_owed or 0.0),
            "priority": priority,
            "evidence": (evidence or "").strip() or None,
        }

    def list_alerts(self) -> list[dict[str, Any]]:
        status_store = VehicleStatusStore(self.database_path)
        obligation_store = VehicleObligationStore(self.database_path)
        by_plate: dict[str, dict[str, Any]] = {}

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT plate_text, status, reason, updated_at FROM vehicle_status"
            ).fetchall()

        for row in rows:
            plate_text = normalize_plate_text(row["plate_text"])
            if not plate_text:
                continue
            amount_owed = obligation_store.outstanding_total(plate_text)
            alert = self._build_alert(plate_text, row["status"], row["reason"], amount_owed, "status_record")
            if alert:
                by_plate[plate_text] = alert

        for plate_text, amount_owed in self._obligation_only_alerts():
            if plate_text in by_plate:
                continue
            alert = self._build_alert(plate_text, "clear", "Outstanding balance due", amount_owed, "obligation")
            if alert:
                by_plate[plate_text] = alert

        alerts = list(by_plate.values())
        alerts.sort(key=lambda item: (
            0 if item["priority"] == "high" else 1 if item["priority"] == "medium" else 2,
            -(float(item["amount_owed"]) or 0.0),
            item["plate_text"],
        ))
        return alerts

    def _obligation_only_alerts(self) -> list[tuple[str, float]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT plate_text, SUM(amount) AS total_amount
                FROM vehicle_obligations
                WHERE status NOT IN ('PAID', 'CANCELLED')
                GROUP BY plate_text
                HAVING SUM(amount) > 0
                """
            ).fetchall()
        return [(normalize_plate_text(str(row["plate_text"])), float(row["total_amount"] or 0.0)) for row in rows if normalize_plate_text(str(row["plate_text"]))]

    def summary(self) -> dict[str, Any]:
        alerts = self.list_alerts()
        summary = {"high": 0, "medium": 0, "low": 0}
        for alert in alerts:
            priority = str(alert.get("priority") or "low")
            summary[priority] = summary.get(priority, 0) + 1
        return {
            "count": len(alerts),
            "high": summary.get("high", 0),
            "medium": summary.get("medium", 0),
            "low": summary.get("low", 0),
            "results": alerts[:10],
        }


class AdminStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS admins (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _hash_password(password: str, salt: bytes | None = None) -> str:
        salt = salt or secrets.token_bytes(16)
        derived = hashlib.scrypt(
            password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=64
        )
        return f"scrypt${base64.urlsafe_b64encode(salt).decode()}${base64.urlsafe_b64encode(derived).decode()}"

    @classmethod
    def _verify_password(cls, password: str, stored_hash: str) -> bool:
        try:
            algorithm, encoded_salt, encoded_derived = stored_hash.split("$", 2)
            if algorithm != "scrypt":
                return False
            salt = base64.urlsafe_b64decode(encoded_salt.encode())
            expected = base64.urlsafe_b64decode(encoded_derived.encode())
            actual = hashlib.scrypt(
                password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=len(expected)
            )
            return hmac.compare_digest(actual, expected)
        except (ValueError, TypeError):
            return False

    def create(self, name: str, email: str, password: str) -> dict[str, Any]:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO admins (name, email, password_hash, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (name, email.lower(), self._hash_password(password), datetime.now(timezone.utc).isoformat()),
            )
            return {"id": cursor.lastrowid, "name": name, "email": email.lower()}

    def authenticate(self, email: str, password: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, name, email, password_hash FROM admins WHERE email = ?",
                (email.lower(),),
            ).fetchone()
        if row is None or not self._verify_password(password, row["password_hash"]):
            return None
        return {"id": row["id"], "name": row["name"], "email": row["email"]}

    def get(self, admin_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, name, email FROM admins WHERE id = ?", (admin_id,)
            ).fetchone()
        return dict(row) if row else None