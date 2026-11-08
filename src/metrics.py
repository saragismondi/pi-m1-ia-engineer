"""Registro de métricas: una línea JSON por request al LLM, append-only, en data/metrics.jsonl.

Costo estimado (USD) = prompt_tokens / 1_000_000 * precio_entrada
                     + completion_tokens / 1_000_000 * precio_salida
Cualquiera puede rehacer la cuenta con los tokens de cada línea y la tabla PRICES_PER_MTOK.
"""
import json
import os
import threading
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

from src.providers import LLMResult

DEFAULT_METRICS_PATH = Path(__file__).resolve().parent.parent / "data" / "metrics.jsonl"

# USD por 1 millón de tokens: (entrada, salida).
# - Anthropic: platform.claude.com/docs/en/about-claude/pricing, consultado el 07/11/2026.
#   Fila "Claude Haiku 5.5 (for prompts up to 100,000 tokens)": $0.10 entrada, $0.50 salida.
# - Groq: usamos el plan gratuito, así que el costo real es 0. (El 07/11/2026 la API de Groq
#   no habilitó llama-3.3-70b-versatile para esta cuenta: se lista con client.models.list()).
PRICES_PER_MTOK = {
    "claude-haiku-5-5": (0.10, 0.50),
    "openai/gpt-oss-20b": (0.0, 0.0),
    "openai/gpt-oss-120b": (0.0, 0.0),  # en uso desde el 07/11/2026: el cupo diario de 20b se agotó en los evals
}


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Devuelve None si el modelo no está en la tabla: preferimos 'sin dato' a inventar un número."""
    if model not in PRICES_PER_MTOK:
        return None
    price_in, price_out = PRICES_PER_MTOK[model]
    return round(prompt_tokens / 1_000_000 * price_in + completion_tokens / 1_000_000 * price_out, 8)


def metrics_path() -> Path:
    """Ruta del log. METRICS_PATH permite apuntar a otro archivo (por ejemplo, en pruebas)."""
    return Path(os.getenv("METRICS_PATH", DEFAULT_METRICS_PATH))


# Identificador del ticket en curso (mejora B1, 08/11/2026). Todas las líneas que se escriben mientras se
# procesa un ticket llevan el mismo ticket_id, así se puede medir la latencia y el costo POR TICKET y no solo
# por llamada. Es un ContextVar: si una llamada corre en otro hilo, hay que copiar el contexto (ver service.py).
_TICKET_ID: ContextVar[str | None] = ContextVar("ticket_id", default=None)


# Un hilo escribe por vez (B2). Sin esto, con el moderador en paralelo, dos líneas se mezclaron en el
# archivo y quedaron JSON rotos (08/11/2026: tests/test_service.py falló en 7 de 15 corridas).
_WRITE_LOCK = threading.Lock()


def new_ticket():
    """Empieza un ticket nuevo: las líneas siguientes llevan un identificador nuevo.

    Devuelve un token para end_ticket(): al terminar, el identificador vuelve a su valor anterior y las
    líneas que se escriban después no quedan asociadas a este ticket.
    """
    return _TICKET_ID.set(uuid.uuid4().hex[:12])


def end_ticket(token) -> None:
    _TICKET_ID.reset(token)


def _append(record: dict) -> dict:
    record["ticket_id"] = _TICKET_ID.get()
    path = metrics_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _WRITE_LOCK, path.open("a", encoding="utf-8") as f:  # "a" = append: nunca se borra lo anterior
        f.write(line)
    return record


def log_event(event: str, provider: str, model: str | None, prompt_version: str,
              category: str | None, latency_s: float) -> dict:
    """Ejecución que NO llamó al LLM: respuesta desde la caché o consulta bloqueada.

    La consigna pide que cada ejecución registre tokens, latencia y costo. Acá los tokens y el
    costo son 0 porque no hubo llamada; la latencia es la del servicio completo.
    event: "cache" | "bloqueado_patron" | "bloqueado_moderador" | "ticket" (fin de un ticket con respuesta;
    la latencia es la de punta a punta, y los tokens de sus llamadas están en las líneas "llm").
    """
    return _append({
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": event,
        "provider": provider,
        "model": model,
        "prompt_version": prompt_version,
        "attempt": 0,
        "valid_contract": True,
        "category": category,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "latency_s": round(latency_s, 3),
        "stop_reason": None,
        "cost_usd": 0.0,
    })


def log_request(result: LLMResult, prompt_version: str, attempt: int, valid: bool, category: str | None) -> dict:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": "llm",
        "provider": result.provider,
        "model": result.model,
        "prompt_version": prompt_version,
        "attempt": attempt,
        "valid_contract": valid,
        "category": category,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.prompt_tokens + result.completion_tokens,
        "latency_s": round(result.latency_s, 3),
        "stop_reason": result.stop_reason,
        "cost_usd": estimate_cost_usd(result.model, result.prompt_tokens, result.completion_tokens),
    }
    return _append(record)
