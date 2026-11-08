"""Tests del ocultamiento de datos personales (src/pii.py, mejora A4). No llaman a ningún LLM: costo 0.

Uso (sin dependencias extra):
    python tests/test_pii.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # permite correrlo como `python tests/test_pii.py`

from src import service  # noqa: E402
from src.pii import redact  # noqa: E402
from src.providers import LLMResult  # noqa: E402

DATASET = [json.loads(l) for l in (ROOT / "evals" / "dataset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def test_cada_tipo_de_dato_se_oculta():
    # Datos de prueba: tarjeta de prueba de la industria (4111...), CUIT con dígito verificador válido.
    casos = {
        "EMAIL": ("Mi mail es juan.perez@empresa.com.ar", "Mi mail es [EMAIL]"),
        "TARJETA": ("Pagué con 4111 1111 1111 1111", "Pagué con [TARJETA]"),
        "CUIT": ("Mi CUIT es 20-12345678-6", "Mi CUIT es [CUIT]"),
        "CBU": ("CBU 0170099220000067797370", "CBU [CBU]"),
        "DNI": ("Mi DNI 30.123.456 no coincide", "Mi DNI [DNI] no coincide"),
        "TELEFONO": ("Llamame al +54 9 11 4444-5555", "Llamame al [TELEFONO]"),
    }
    for kind, (text, expected) in casos.items():
        out, counts = redact(text)
        assert out == expected and counts == {kind: 1}, f"{kind}: {out!r} {counts}"


def test_telefono_con_palabra_conserva_la_palabra():
    out, counts = redact("escribime al cel 11 4444 5555")
    assert out == "escribime al cel [TELEFONO]" and counts == {"TELEFONO": 1}, out


def test_numeros_de_negocio_no_se_ocultan():
    # Controles de falsos positivos: facturas, montos, horas, errores, planes y tarjetas inválidas.
    textos = [
        "La factura 0001-00012345 de $15.000 tiene 3 ítems",
        "Me da error 500 desde las 14:30",
        "El plan 10 dice 50 usuarios",
        "Pagué con 4111 1111 1111 1112",  # no pasa Luhn
        "Mi CUIT es 20-12345678-5",  # dígito verificador incorrecto
        "El pedido 12345678 no llegó",  # 8 dígitos sin la palabra DNI
    ]
    for text in textos:
        out, counts = redact(text)
        assert out == text and counts == {}, f"{text!r} -> {out!r}"


def test_el_dataset_no_cambia():
    # Los 40 tickets reales no tienen datos personales: ninguno debe modificarse.
    for case in DATASET:
        out, counts = redact(case["query"])
        assert out == case["query"], f"{case['id']}: {counts}"


class RecordingProvider:
    """Proveedor falso que guarda lo que recibe, para comprobar que la IA no ve los datos."""
    name, model = "fake", "fake-model"

    def __init__(self):
        self.seen = []

    def complete(self, system, user, schema=None):
        self.seen.append(user)
        payload = {"answer": "Vamos a revisar tu caso.", "confidence": 0.9, "category": "billing",
                   "actions": ["Revisar el cobro."]}
        return LLMResult(text=json.dumps(payload), provider=self.name, model=self.model,
                         prompt_tokens=10, completion_tokens=10, latency_s=0.0, stop_reason="end_turn")


def _run(query: str, pii_env: str | None):
    fake = RecordingProvider()
    original = service.get_provider
    service.get_provider = lambda name: fake
    saved = {k: os.environ.get(k) for k in ("METRICS_PATH", "PII_REDACTION")}
    with tempfile.TemporaryDirectory() as d:
        os.environ["METRICS_PATH"] = str(Path(d) / "m.jsonl")
        if pii_env is None:
            os.environ.pop("PII_REDACTION", None)
        else:
            os.environ["PII_REDACTION"] = pii_env
        try:
            result = service.triage(query, "fake", use_cache=False, moderation="off")
        finally:
            service.get_provider = original
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    return result, fake.seen


def test_la_ia_no_recibe_los_datos():
    result, seen = _run("me cobraron dos veces, mi mail es ana@empresa.com y mi tarjeta 4111 1111 1111 1111", None)
    assert len(seen) == 1
    assert "ana@empresa.com" not in seen[0] and "4111" not in seen[0], seen[0]
    assert "[EMAIL]" in seen[0] and "[TARJETA]" in seen[0]
    assert result.pii_redacted == {"EMAIL": 1, "TARJETA": 1}


def test_se_puede_apagar():
    result, seen = _run("mi mail es ana@empresa.com", "off")
    assert "ana@empresa.com" in seen[0] and result.pii_redacted == {}


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
