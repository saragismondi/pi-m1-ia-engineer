"""Oculta datos personales antes de mandar el ticket a un proveedor de IA (mejora A4, 08/11/2026).

Por qué: el ticket viaja a servidores de terceros (Anthropic, Groq). Mandar lo mínimo necesario es
práctica general de la industria. Para clasificar y proponer una respuesta, la IA no necesita el email,
el CUIT ni el número de tarjeta: le alcanza con saber que el cliente los mencionó.

Qué se oculta (cada dato se reemplaza por un marcador, por ejemplo "[EMAIL]"):
  - emails;
  - tarjetas: de 13 a 19 dígitos que pasan el algoritmo de Luhn (así no se tapa cualquier número largo);
  - CUIT/CUIL: 11 dígitos con prefijo válido y dígito verificador correcto;
  - CBU: 22 dígitos seguidos;
  - DNI: solo si viene después de la palabra "DNI" o "documento" (un número de 8 dígitos solo puede ser
    una factura o un pedido);
  - teléfonos: solo con "+" adelante o después de "tel", "cel", "celular", "teléfono" o "whatsapp".

Qué NO cambia: el agente humano sigue viendo el ticket original. Solo cambia lo que recibe la IA.
Limitación: son patrones. Un dato escrito de forma rara (por ejemplo "mi mail es juan arroba gmail")
no se detecta.
"""
import re

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_CARD = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")
_CUIT = re.compile(r"\b(20|23|24|27|30|33|34)-?(\d{8})-?(\d)\b")
_CBU = re.compile(r"\b\d{22}\b")
_DNI = re.compile(r"\b(dni|d\.n\.i\.|documento)(\s*(n(ro|°|º)?\.?|numero|número))?\s*:?\s*(\d{1,2}\.?\d{3}\.?\d{3})\b",
                  re.IGNORECASE)
_PHONE = re.compile(
    r"\+\d[\d\s-]{7,16}\d"
    r"|\b(tel|tel\.|telefono|teléfono|cel|celular|whatsapp|wsp)\s*:?\s*\(?\d[\d\s()-]{6,16}\d",
    re.IGNORECASE)


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _cuit_ok(prefix: str, body: str, check: str) -> bool:
    digits = [int(c) for c in prefix + body]
    weights = [5, 4, 3, 2, 7, 6, 5, 4, 3, 2]
    rest = 11 - sum(d * w for d, w in zip(digits, weights)) % 11
    expected = 0 if rest == 11 else 9 if rest == 10 else rest
    return expected == int(check)


def redact(text: str) -> tuple[str, dict[str, int]]:
    """Devuelve (texto con los datos ocultos, cuántos datos de cada tipo se ocultaron)."""
    counts: dict[str, int] = {}

    def sub(pattern: re.Pattern, label: str, valid=None):
        nonlocal text

        def repl(m: re.Match) -> str:
            if valid and not valid(m):
                return m.group(0)
            counts[label] = counts.get(label, 0) + 1
            if label in ("DNI", "TELEFONO") and m.lastindex and m.group(1) and not m.group(0).startswith("+"):
                # Se conserva la palabra ("DNI", "cel") y se oculta solo el número.
                return f"{m.group(1)} [{label}]"
            return f"[{label}]"

        text = pattern.sub(repl, text)

    sub(_EMAIL, "EMAIL")
    sub(_CUIT, "CUIT", lambda m: _cuit_ok(m.group(1), m.group(2), m.group(3)))
    sub(_CBU, "CBU")
    sub(_CARD, "TARJETA", lambda m: _luhn_ok(re.sub(r"\D", "", m.group(0))))
    sub(_DNI, "DNI")
    sub(_PHONE, "TELEFONO")
    return text, counts
