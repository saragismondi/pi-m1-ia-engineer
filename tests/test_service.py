"""Tests del parseo del JSON y de los reintentos (src/service.py, mejora D1). Sin LLM: costo 0.

Uso (sin dependencias extra):
    python tests/test_service.py

Usan un proveedor falso que devuelve respuestas armadas a mano, en orden. Así se puede probar qué
pasa cuando el modelo responde mal, sin gastar llamadas reales y sin depender de que el modelo falle.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # permite correrlo como `python tests/test_service.py`

import httpx2  # noqa: E402  (el cliente HTTP que usa openai 3.46.0)
import openai  # noqa: E402

from src import service  # noqa: E402
from src.prompts import load_prompt  # noqa: E402
from src.providers import LLMResult  # noqa: E402

VALID = {"answer": "Vamos a revisar tu caso.", "confidence": 0.9, "category": "billing",
         "actions": ["Revisar el historial de pagos."]}
PROMPT = load_prompt("v1")


class ScriptedProvider:
    """Devuelve, en orden, las respuestas del guion. Un elemento Exception se lanza en vez de devolverse."""
    name, model = "fake", "fake-model"

    def __init__(self, script):
        self.script = list(script)
        self.systems = []  # el system prompt que recibió en cada llamada

    def complete(self, system, user, schema=None):
        self.systems.append(system)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        text, stop = item if isinstance(item, tuple) else (item, "end_turn")
        return LLMResult(text=text, provider=self.name, model=self.model,
                         prompt_tokens=10, completion_tokens=10, latency_s=0.0, stop_reason=stop)


def run(provider):
    """call_llm con las métricas en un archivo temporal. Devuelve (respuesta o excepción, líneas de métricas)."""
    saved = os.environ.get("METRICS_PATH")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.jsonl"
        os.environ["METRICS_PATH"] = str(path)
        try:
            outcome = service.call_llm(provider, PROMPT, "me cobraron dos veces")
        except service.TriageError as e:
            outcome = e
        finally:
            if saved is None:
                os.environ.pop("METRICS_PATH", None)
            else:
                os.environ["METRICS_PATH"] = saved
        lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return outcome, lines


# --- Parseo -----------------------------------------------------------------

def test_json_limpio():
    assert service.parse_response(json.dumps(VALID)).category == "billing"


def test_json_envuelto_en_markdown_o_con_texto():
    for text in ("```json\n" + json.dumps(VALID) + "\n```", "Acá va: " + json.dumps(VALID) + " ¡Saludos!"):
        assert service.parse_response(text).category == "billing", text


def test_sin_json_es_error():
    for text in ("", "No puedo ayudarte con eso.", "{ incompleto"):
        try:
            service.parse_response(text)
        except ValueError:
            continue
        raise AssertionError(f"{text!r} debía fallar")


def test_contrato_invalido_explica_el_campo():
    bad = dict(VALID, category="ventas", confidence=1.5)
    try:
        service.parse_response(json.dumps(bad))
    except Exception as e:  # noqa: BLE001
        msg = service.describe_error(e)
        assert "category" in msg and "confidence" in msg, msg
    else:
        raise AssertionError("debía fallar")


# --- Reintentos -------------------------------------------------------------

def test_primera_respuesta_valida_no_reintenta():
    provider = ScriptedProvider([json.dumps(VALID)])
    outcome, lines = run(provider)
    assert outcome.category == "billing" and len(provider.systems) == 1
    assert [(l["attempt"], l["valid_contract"]) for l in lines] == [(1, True)]


def test_reintenta_y_pasa_el_error_al_modelo():
    bad = json.dumps(dict(VALID, category="ventas"))
    provider = ScriptedProvider([bad, "sin json", json.dumps(VALID)])
    outcome, lines = run(provider)
    assert outcome.category == "billing"
    assert [(l["attempt"], l["valid_contract"]) for l in lines] == [(1, False), (2, False), (3, True)]
    assert provider.systems[0] == PROMPT.system  # el primer intento usa el prompt tal cual
    assert "category" in provider.systems[1] and "no respetó el contrato" in provider.systems[1]
    assert "no contiene un objeto JSON" in provider.systems[2]


def test_tres_fallos_terminan_en_error():
    provider = ScriptedProvider(["nada", "nada", "nada"])
    outcome, lines = run(provider)
    assert isinstance(outcome, service.TriageError) and "3 intentos" in str(outcome)
    assert len(provider.systems) == 3 and [l["attempt"] for l in lines] == [1, 2, 3]


def test_respuesta_cortada_avisa_max_tokens():
    provider = ScriptedProvider([('{"answer": "cortad', "max_tokens"), json.dumps(VALID)])
    outcome, _ = run(provider)
    assert outcome.category == "billing"
    assert "se cortó por MAX_TOKENS" in provider.systems[1]


def test_error_de_api_no_se_reintenta_en_el_servicio():
    # Los errores de red y de límites los reintenta el SDK (SDK_MAX_RETRIES); el servicio no los repite.
    error = openai.APIConnectionError(request=httpx2.Request("POST", "https://api.test"))
    provider = ScriptedProvider([error, json.dumps(VALID)])
    outcome, lines = run(provider)
    assert isinstance(outcome, service.TriageError) and "APIConnectionError" in str(outcome)
    assert len(provider.systems) == 1 and lines == []


# --- Métricas por ticket (mejora B1) ----------------------------------------

SCORES_OK_IN = json.dumps({"inyeccion": 0.0, "jailbreak": 0.0, "extraccion": 0.0})
SCORES_OK_OUT = json.dumps({"promesa_indebida": 0.0, "fuga_interna": 0.0, "obedece_inyeccion": 0.0})


def _triage_lines(queries):
    """Corre triage() completo (moderador de entrada, generación, moderador de salida) con un proveedor falso."""
    script = []
    for _ in queries:
        script += [SCORES_OK_IN, json.dumps(VALID), SCORES_OK_OUT]
    provider = ScriptedProvider(script)
    original = service.get_provider
    service.get_provider = lambda name: provider
    saved = {k: os.environ.get(k) for k in ("METRICS_PATH", "MODERATION_PARALLEL")}
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.jsonl"
        os.environ["METRICS_PATH"] = str(path)
        os.environ["MODERATION_PARALLEL"] = "off"  # el guion es en orden: en paralelo el orden no es fijo
        try:
            for q in queries:
                service.triage(q, "fake", use_cache=False, moderation="both")
        finally:
            service.get_provider = original
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def test_todas_las_lineas_de_un_ticket_comparten_id():
    lines = _triage_lines(["me cobraron dos veces"])
    assert [l["prompt_version"] for l in lines] == ["moderador_entrada", "v1", "moderador_salida", "v1"]
    assert lines[-1]["event"] == "ticket"
    ids = {l["ticket_id"] for l in lines}
    assert len(ids) == 1 and None not in ids, ids


def test_cada_ticket_tiene_su_propio_id():
    lines = _triage_lines(["me cobraron dos veces", "no puedo entrar"])
    tickets = [l for l in lines if l["event"] == "ticket"]
    assert len(tickets) == 2 and tickets[0]["ticket_id"] != tickets[1]["ticket_id"]


# --- Moderador en paralelo (mejora B2) ---------------------------------------

class SchemaAwareProvider:
    """Responde según el esquema pedido (moderador de entrada, de salida o triage) y tarda `delay` s.

    Hace falta porque en paralelo el orden de las llamadas no es fijo: un guion en orden no sirve.
    """
    name, model = "fake", "fake-model"

    def __init__(self, delay=0.0, flag_input=False, fail_generation=False, moderator_api_error=False,
                 generation_bug=False):
        self.delay, self.flag_input, self.fail_generation = delay, flag_input, fail_generation
        self.moderator_api_error, self.generation_bug = moderator_api_error, generation_bug

    def complete(self, system, user, schema=None):
        import time
        time.sleep(self.delay)
        props = (schema or {}).get("properties", {})
        if "inyeccion" in props and self.moderator_api_error:
            raise openai.APIConnectionError(request=httpx2.Request("POST", "https://api.test"))
        if "inyeccion" not in props and "promesa_indebida" not in props and self.generation_bug:
            raise RuntimeError("bug en la generación")
        if "inyeccion" in props:
            text = json.dumps({"inyeccion": 0.9 if self.flag_input else 0.0, "jailbreak": 0.0, "extraccion": 0.0})
        elif "promesa_indebida" in props:
            text = SCORES_OK_OUT
        elif self.fail_generation:
            text = "sin json"
        else:
            text = json.dumps(VALID)
        return LLMResult(text=text, provider=self.name, model=self.model,
                         prompt_tokens=10, completion_tokens=10, latency_s=self.delay, stop_reason="end_turn")


def _run_triage(provider, parallel: str):
    import time
    original = service.get_provider
    service.get_provider = lambda name: provider
    saved = {k: os.environ.get(k) for k in ("METRICS_PATH", "MODERATION_PARALLEL")}
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.jsonl"
        os.environ["METRICS_PATH"] = str(path)
        os.environ["MODERATION_PARALLEL"] = parallel
        start = time.perf_counter()
        try:
            outcome = service.triage("me cobraron dos veces", "fake", use_cache=False, moderation="both")
        except Exception as e:  # noqa: BLE001  (TriageError o un bug que debe subir)
            outcome = e
        finally:
            elapsed = time.perf_counter() - start
            service.get_provider = original
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    return outcome, lines, elapsed


def test_paralelo_da_el_mismo_resultado_y_mismo_ticket_id():
    outcome, lines, _ = _run_triage(SchemaAwareProvider(), "on")
    assert outcome.response.category == "billing"
    assert sorted(l["prompt_version"] for l in lines) == ["moderador_entrada", "moderador_salida", "v1", "v1"]
    assert len({l["ticket_id"] for l in lines}) == 1


def test_paralelo_es_mas_rapido():
    # Cada llamada tarda 0,5 s. En serie: 3 llamadas = ~1,5 s. En paralelo: ~1,0 s. Umbral: 0,2 s.
    _, _, serial = _run_triage(SchemaAwareProvider(delay=0.5), "off")
    _, _, parallel = _run_triage(SchemaAwareProvider(delay=0.5), "on")
    assert serial - parallel > 0.2, f"serie {serial:.2f} s, paralelo {parallel:.2f} s"


def test_paralelo_bloquea_igual_y_la_generacion_se_paga():
    outcome, lines, _ = _run_triage(SchemaAwareProvider(flag_input=True), "on")
    assert outcome.input_flag and outcome.input_flag.startswith("moderador:")
    assert outcome.response.category == "other"
    events = [(l["event"], l["prompt_version"]) for l in lines]
    assert ("llm", "v1") in events  # la respuesta generada se pagó aunque se descartó
    assert events[-1] == ("bloqueado_moderador", "v1")  # nada se escribe después del bloqueo


def test_paralelo_error_de_generacion_llega_igual():
    outcome, _, _ = _run_triage(SchemaAwareProvider(fail_generation=True), "on")
    assert isinstance(outcome, service.TriageError) and "3 intentos" in str(outcome)


def test_paralelo_moderador_con_error_de_api_bloquea():
    # Falla cerrada, igual que en serie: si el moderador no responde, el ticket va a una persona.
    outcome, lines, _ = _run_triage(SchemaAwareProvider(moderator_api_error=True), "on")
    assert outcome.input_flag and "no_disponible" in outcome.input_flag, outcome.input_flag


def test_paralelo_un_bug_de_la_generacion_no_se_esconde_detras_del_bloqueo():
    outcome, _, _ = _run_triage(SchemaAwareProvider(flag_input=True, generation_bug=True), "on")
    assert isinstance(outcome, RuntimeError), outcome


def test_el_ticket_id_se_limpia_al_terminar():
    from src.metrics import _TICKET_ID
    before = _TICKET_ID.get()
    _run_triage(SchemaAwareProvider(), "off")
    assert _TICKET_ID.get() == before


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
