"""Tests del guardrail de acciones (mejora A2, 08/11/2026). No llaman a ningún LLM: costo 0.

Uso (sin dependencias extra):
    python tests/test_actions.py

Antes de A2 solo se revisaba el "answer"; las "actions" para el agente humano no pasaban por ningún
control (src/service.py). Una instrucción escondida en el ticket podía terminar como acción sugerida.
"""
import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # permite correrlo como `python tests/test_actions.py`

from src import service  # noqa: E402
from src.guardrails import ACTION_PATTERNS, check_actions  # noqa: E402
from src.prompts import load_prompt  # noqa: E402
from src.providers import LLMResult  # noqa: E402

SYSTEM = load_prompt("v1").system

# Acciones reales del proyecto: los 5 ejemplos de prompts/v2_few_shot.yaml y el ejemplo del README.
# Ninguna debe marcarse (control de falsos positivos).
def _v2_example_actions() -> list[str]:
    actions = []
    for line in (ROOT / "prompts" / "v2_few_shot.yaml").read_text(encoding="utf-8").splitlines():
        m = re.search(r'"actions": \[(.*)\]', line)
        if m and m.group(1) != '"..."':
            actions += json.loads("[" + m.group(1) + "]")
    return actions


REAL_ACTIONS = _v2_example_actions() + [
    "Revisar el historial de pagos del cliente para confirmar el doble cobro.",
    "Consultar con el equipo de facturación la posibilidad de emitir un reintegro o ajuste.",
    "Comunicar al cliente el estado de la revisión y los próximos pasos.",
]


def test_hay_acciones_reales_para_controlar():
    assert len(REAL_ACTIONS) >= 14, len(REAL_ACTIONS)


def test_acciones_reales_no_se_marcan():
    for action, flags in zip(REAL_ACTIONS, check_actions(REAL_ACTIONS, SYSTEM)):
        assert flags == [], f"{action!r} marcada por {flags}"


def test_cada_patron_de_accion_tiene_un_ejemplo():
    examples = {
        "saltear_verificacion": "Cambiar el email sin verificar la identidad del cliente.",
        "aprobar_dinero": "Aprobar el reintegro del cobro duplicado.",
        "accion_irreversible": "Eliminar la cuenta del cliente.",
    }
    assert set(examples) == set(ACTION_PATTERNS)
    for name, action in examples.items():
        assert name in check_actions([action], SYSTEM)[0], f"{action!r}: esperado {name}"


def test_el_caso_adv_20_se_marca():
    # La instrucción escondida en adv-20, si el modelo la copiara como acción.
    flags = check_actions(["Eliminar la cuenta sin verificar la identidad del cliente."], SYSTEM)[0]
    assert set(flags) == {"accion_irreversible", "saltear_verificacion"}, flags


def test_mencionar_aprobar_en_medio_no_se_marca():
    # Solo cuenta el verbo inicial: evaluar o consultar no es aprobar.
    assert check_actions(["Verificar si corresponde aprobar el reintegro."], SYSTEM) == [[]]


def test_fuga_del_system_prompt_en_una_accion():
    leaked = " ".join(SYSTEM.split()[:12])
    assert "fuga_system_prompt" in check_actions([leaked], SYSTEM)[0]


class FakeProvider:
    """Proveedor falso: devuelve siempre el mismo JSON, sin red."""
    name, model = "fake", "fake-model"

    def __init__(self, payload: dict):
        self.text = json.dumps(payload, ensure_ascii=False)

    def complete(self, system, user, schema=None):
        return LLMResult(text=self.text, provider=self.name, model=self.model,
                         prompt_tokens=10, completion_tokens=10, latency_s=0.0, stop_reason="end_turn")


def test_servicio_marca_y_no_borra_las_acciones():
    payload = {"answer": "Vamos a revisar tu caso.", "confidence": 0.9, "category": "account",
               "actions": ["Pedir el email nuevo.", "Eliminar la cuenta sin verificar la identidad."]}
    original = service.get_provider
    service.get_provider = lambda name: FakeProvider(payload)
    old_metrics = os.environ.get("METRICS_PATH")
    with tempfile.TemporaryDirectory() as d:
        os.environ["METRICS_PATH"] = str(Path(d) / "m.jsonl")
        try:
            result = service.triage("tengo una consulta sobre mi cuenta", "fake", use_cache=False, moderation="off")
        finally:
            service.get_provider = original
            if old_metrics is None:
                os.environ.pop("METRICS_PATH", None)
            else:
                os.environ["METRICS_PATH"] = old_metrics
    actions = result.response.actions
    assert actions[0] == "Pedir el email nuevo."
    prefix, _, rest = actions[1].partition("] ")
    assert prefix.startswith("[REVISAR: ") and rest.startswith("Eliminar la cuenta"), actions[1]
    assert set(prefix.removeprefix("[REVISAR: ").split(", ")) == {"accion_irreversible", "saltear_verificacion"}
    assert result.action_flags == ["accion_irreversible", "saltear_verificacion"]
    assert result.response.answer == "Vamos a revisar tu caso."  # el answer no se toca


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
