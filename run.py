import os

import uvicorn


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload_enabled = _to_bool(os.getenv("RELOAD"), default=False)

    uvicorn.run("app.app:app", host=host, port=port, reload=reload_enabled)