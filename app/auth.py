import base64
import hashlib
import hmac
import json
import time
from typing import Any

from fastapi import HTTPException, Request


def create_session(admin_id: int, secret: str, days: int) -> str:
    payload = {"admin_id": admin_id, "expires": int(time.time()) + days * 86400}
    encoded = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()
    signature = hmac.new(secret.encode(), encoded.encode(), hashlib.sha256).hexdigest()
    return f"{encoded}.{signature}"


def read_session(value: str | None, secret: str) -> dict[str, Any] | None:
    if not value or "." not in value:
        return None
    encoded, signature = value.rsplit(".", 1)
    expected = hmac.new(secret.encode(), encoded.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        payload = json.loads(base64.urlsafe_b64decode(encoded.encode()))
        if int(payload["expires"]) < int(time.time()):
            return None
        return payload
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def require_admin(request: Request, admin_store: Any, secret: str) -> dict[str, Any]:
    session = read_session(request.cookies.get("vnpr_session"), secret)
    admin = admin_store.get(int(session["admin_id"])) if session else None
    if admin is None:
        raise HTTPException(status_code=401, detail="Sign in as an admin to continue")
    return admin