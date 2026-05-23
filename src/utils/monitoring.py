def capture_event(name: str, payload: dict) -> dict:
    return {"event": name, "payload": payload}
