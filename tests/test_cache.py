"""Tests de la caché (src/cache.py). No llaman a ningún LLM: costo 0.

Uso (sin dependencias extra):
    python tests/test_cache.py

Mejora A5 (08/11/2026): el modo de moderación está en la clave. Antes, una respuesta guardada con
--moderation off se reutilizaba con la moderación prendida y salteaba al moderador.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # permite correrlo como `python tests/test_cache.py`

from src import cache, service  # noqa: E402
from src.providers import LLMResult  # noqa: E402

PAYLOAD = {"answer": "Vamos a revisar tu caso.", "confidence": 0.9, "category": "billing",
           "actions": ["Revisar el historial de pagos."]}


class FakeProvider:
    """Proveedor falso: cuenta las llamadas y devuelve siempre el mismo JSON, sin red."""
    name, model = "fake", "fake-model"

    def __init__(self):
        self.calls = 0

    def complete(self, system, user, schema=None):
        self.calls += 1
        return LLMResult(text=json.dumps(PAYLOAD), provider=self.name, model=self.model,
                         prompt_tokens=10, completion_tokens=10, latency_s=0.0, stop_reason="end_turn")


def test_la_clave_cambia_con_el_modo_de_moderacion():
    keys = {cache.make_key("p", "m", "v1", mode, "hola") for mode in ("off", "output", "both")}
    assert len(keys) == 3


def test_mayusculas_y_espacios_no_cambian_la_clave():
    a = cache.make_key("p", "m", "v1", "both", "Me cobraron  dos veces")
    b = cache.make_key("p", "m", "v1", "both", "me cobraron dos veces")
    assert a == b


def test_respuesta_guardada_sin_moderacion_no_se_usa_con_moderacion():
    fake = FakeProvider()
    original = service.get_provider
    service.get_provider = lambda name: fake
    saved = {k: os.environ.get(k) for k in ("CACHE_PATH", "METRICS_PATH")}
    with tempfile.TemporaryDirectory() as d:
        os.environ["CACHE_PATH"] = str(Path(d) / "c.jsonl")
        os.environ["METRICS_PATH"] = str(Path(d) / "m.jsonl")
        try:
            first = service.triage("me cobraron dos veces", "fake", moderation="off")
            again = service.triage("me cobraron dos veces", "fake", moderation="off")
            moderated = service.triage("me cobraron dos veces", "fake", moderation="both")
        finally:
            service.get_provider = original
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    assert not first.cached and again.cached  # misma consulta y mismo modo: sale de la caché
    assert not moderated.cached  # con otro modo no se reutiliza: pasa por el moderador


def main() -> int:
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FALLA {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests ok")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
