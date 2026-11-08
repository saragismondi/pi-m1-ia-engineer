"""Caché de respuestas con clave MD5 (extra credit).

Si llega la misma consulta con el mismo proveedor, modelo, versión de prompt y modo de moderación,
devolvemos la respuesta guardada sin volver a llamar al LLM: costo 0 y latencia casi 0.

Clave = md5(proveedor | modelo | versión de prompt | modo de moderación | consulta normalizada).

El modo de moderación está en la clave desde el 08/11/2026 (mejora A5). Antes, una respuesta guardada
con la moderación apagada se reutilizaba con la moderación prendida, y se salteaba el moderador.
MD5 no se usa acá por seguridad, solo para tener una clave corta y fija a partir de un texto.

Archivo: data/cache.jsonl, una línea {"key": ..., "response": {...}} por entrada (append-only).
Si una clave aparece dos veces, gana la última.
"""
import hashlib
import json
import os
from pathlib import Path

DEFAULT_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "cache.jsonl"


def cache_path() -> Path:
    return Path(os.getenv("CACHE_PATH", DEFAULT_CACHE_PATH))


def make_key(provider: str, model: str, prompt_version: str, moderation: str, query: str) -> str:
    normalized = " ".join(query.lower().split())  # mayúsculas y espacios extra no cambian la clave
    raw = f"{provider}|{model}|{prompt_version}|{moderation}|{normalized}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def get(key: str) -> dict | None:
    path = cache_path()
    if not path.exists():
        return None
    found = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            if entry["key"] == key:
                found = entry["response"]
    return found


def put(key: str, response: dict) -> None:
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "response": response}, ensure_ascii=False) + "\n")
