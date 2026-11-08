"""Guardrails: controles deterministas (sin LLM, costo 0) antes y después del modelo.

Entrada (check_input): rechaza consultas vacías o demasiado largas y detecta intentos
de prompt injection con patrones de texto. Si detecta uno, el servicio NO llama al LLM.

Salida (check_output): revisa el "answer" del modelo antes de mostrarlo:
  - promesas (reintegros, correcciones, plazos) que solo puede decidir un agente humano;
  - fragmentos del system prompt (fuga de instrucciones).
Si encuentra algo, el servicio reemplaza el "answer" por uno seguro.

Limitación conocida: son patrones de texto. Un ataque reformulado puede pasar el guardrail
de entrada; por eso el prompt también trata el ticket como dato y existe el control de salida.
"""
import base64
import binascii
import re
import unicodedata

MAX_QUERY_CHARS = 2000  # la consulta más larga del dataset mide 116 caracteres

# Se aplican sobre el texto normalizado: minúsculas y sin tildes.
INJECTION_PATTERNS = {
    "ignorar_instrucciones": r"\b(ignor|olvid|descart|saltea|omit)\w*\s+(\w+\s+){0,4}(instrucciones|reglas|indicaciones|prompt)",
    "ignore_instructions_en": r"\b(ignore|disregard|forget)\s+(\w+\s+){0,3}(instructions|rules|prompt)",
    "pedir_prompt": r"\b(revela|mostra|mostrame|muestra|decime|dime|imprimi|imprime|repeti|repite|copia|escribi)\w*\s+(\w+\s+){0,5}(instrucciones|prompt|reglas|configuracion)",
    "mencion_system_prompt": r"\b(system prompt|prompt del sistema|prompt de sistema|instrucciones del sistema|instrucciones internas)",
    "cambio_de_rol": r"\b(a partir de ahora|desde ahora)\s+(sos|eres|actua|responde|tenes que)|\b(actua|comportate)\s+como\b|\byou are now\b",
    "modo_sin_reglas": r"\b(modo desarrollador|developer mode|jailbreak|sin restricciones|sin filtros|sin reglas)\b",
    "etiquetas_falsas": r"</?\s*(ticket|system|instrucciones)\s*>|\[/?(system|inst)\]|^\s*(system|assistant)\s*:",
    # Portugués (A1, 08/11/2026): sin tildes, "instruções" queda "instrucoes".
    "ignorar_instrucciones_pt": r"\b(ignore|esqueca|desconsidere|descarte)\w*\s+(\w+\s+){0,4}(instrucoes|regras|prompt)|\b(mostre|revele|repita)\w*\s+(\w+\s+){0,4}(instrucoes|regras|prompt)",
}

# --- Desofuscación (A1, 08/11/2026) ------------------------------------------------------------
# Trucos de escritura que esquivaban los patrones (probados el 07/11/2026 con check_input):
# espacio invisible, letras cirílicas o griegas parecidas, números por letras, letras separadas y base64.
# Los patrones se prueban sobre el texto normal y además sobre estas versiones "desofuscadas".
# El texto que recibe el modelo NO cambia: esto solo sirve para detectar.

# Caracteres de formato invisibles (categoría Unicode "Cf"): espacio de ancho cero, unión de palabras, etc.
_INVISIBLE_CATEGORY = "Cf"

# Letras de otros alfabetos que se ven iguales a las latinas (las más usadas para engañar filtros).
_HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j", "ѕ": "s",
    "ԁ": "d", "һ": "h", "ӏ": "l", "ν": "v", "ο": "o", "α": "a", "ε": "e", "ι": "i", "κ": "k", "ρ": "p",
    "τ": "t", "υ": "u",
})

# Números y símbolos usados como letras ("1gn0rá"). Solo se aplica dentro de palabras que tienen
# letras, para no convertir números sueltos como "1023".
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})

# Base64: bloques largos que, decodificados, son texto legible.
_BASE64_TOKEN = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")

PROMISE_PATTERNS = {
    "reintegro": r"\b(te|le|les)\s+(vamos a |va a |van a )?(devolver|reintegrar|reembolsar)\w*|\b(devolveremos|reintegraremos|reembolsaremos)\b|\b(vamos a|va a|van a)\s+(devolver|reintegrar|reembolsar|anular el cobro)",
    "correccion": r"\b(corregiremos|solucionaremos|resolveremos|arreglaremos|repararemos)\b|\b(vamos a|va a|van a)\s+(corregir|solucionar|resolver|arreglar|reparar)\b",
    "garantia": r"\b(te|le)\s+(garantizo|garantizamos|aseguro|aseguramos)\b|\bgarantizad[oa]\b",
    "plazo": r"\ben\s+(las\s+)?(proximas\s+)?\d+\s*(minutos|horas|hs|dias)\b|\bhoy mismo\b|\bde inmediato\b",
}

# --- Alcance cerrado (S0, 08/11/2026) ------------------------------------------------------------
# Este servicio atiende soporte de Nubbix: facturación, problemas técnicos, cuenta y consultas comerciales.
# Un pedido ajeno (programación general, tareas, recetas, poemas, traducciones, cultura general) no tiene
# que llegar a ninguna IA: se responde con un texto fijo, sin gastar llamadas. Antes de S0 esos pedidos
# gastaban 3 llamadas y dependían de que el modelo decidiera no resolverlos (probado el 08/11/2026).
#
# Para no bloquear tickets legítimos, cada regla exige un PEDIDO DE PRODUCIR algo ajeno, y además el ticket
# no puede mencionar nada de Nubbix (NUBBIX_CONTEXT). "Nuestra integración en Python con la API da error
# 401" menciona la API: no se bloquea. Lo ajeno escrito de otra forma lo evalúa el moderador (fuera_de_alcance).
OUT_OF_SCOPE_PATTERNS = {
    "programacion": r"\b(escrib|program|codific|gener|hac|arm|cre|desarroll|pasa)\w*\s+(\w+\s+){0,4}(funcion|codigo|script|programa|algoritmo|clase en|consulta sql|regex|expresion regular)\b",
    "tarea_matematica": r"\b(resolv|calcul|deriv|integr|demostr)\w*\s+(\w+\s+){0,4}(ejercicio|ecuacion|derivada|integral|teorema|limite de)\b",
    "escritura_creativa": r"\b(escrib|redact|compon|hac|cre|invent)\w*\s+(\w+\s+){0,4}(poema|cuento|cancion|ensayo|chiste|novela|monografia|rap)\b",
    "receta": r"\breceta\s+(de|para)\b|\bcomo\s+(se\s+)?(cocin|prepar|hornea)\w*\s+(un|una|el|la)\b",
    "traduccion": r"\btraduc\w*\s+(\w+\s+){0,6}(al|del)\s+(ingles|portugues|frances|aleman|italiano|chino|japones)\b",
    "cultura_general": r"\bquien\s+(fue|invento|descubrio|gano)\b|\bcual\s+es\s+la\s+capital\s+de\b",
    "tarea_escolar": r"\b(hac|resolv)\w*\s+(mi|la|esta)\s+tarea\b",
}

# Palabras del mundo de Nubbix. Si el ticket menciona alguna, no se bloquea por alcance: puede ser un
# pedido legítimo sobre el producto (lo decide el resto del flujo).
NUBBIX_CONTEXT = r"\b(nubbix|factura\w*|cobr\w*|pag\w*|cuenta|usuario\w*|sistema|plataforma|api|integracion\w*|error\w*|plan|planes|suscripcion\w*|stock|venta\w*|clientes?|afip|reporte\w*|export\w*|contrasena|login|sucursal\w*|soporte|modulo\w*|app|aplicacion)\b"

OUT_OF_SCOPE_ANSWER = ("Este canal atiende consultas sobre Nubbix: facturación, problemas técnicos, tu cuenta y "
                       "nuestros planes. Tu consulta está fuera de ese alcance, así que no podemos ayudarte con "
                       "eso por acá. Si tenés alguna duda sobre Nubbix, escribinos.")


def check_scope(query: str) -> str | None:
    """Nombre de la regla si el pedido es claramente ajeno a Nubbix; None si no (o si menciona Nubbix)."""
    text = normalize(query)
    if re.search(NUBBIX_CONTEXT, text):
        return None
    for name, pattern in OUT_OF_SCOPE_PATTERNS.items():
        if re.search(pattern, text):
            return name
    return None


# Acciones para el agente humano (A2, 08/11/2026). Antes solo se revisaba el "answer": una instrucción
# escondida en el ticket podía terminar como acción sugerida (por ejemplo, adv-20: "eliminar la cuenta
# sin verificar la identidad"). Los patrones se aplican a cada acción normalizada. Los anclados con ^
# solo miran el verbo inicial, para no marcar acciones como "verificar si corresponde aprobar...".
ACTION_PATTERNS = {
    "saltear_verificacion": r"\bsin\s+(verificar|validar|confirmar|chequear|comprobar)\b",
    "aprobar_dinero": r"^(aprobar|otorgar|emitir|procesar|acreditar|confirmar|autorizar)\s+(el\s+|la\s+|un\s+|una\s+)?(reintegro|reembolso|devolucion|bonificacion|descuento|credito)",
    "accion_irreversible": r"^(eliminar|borrar|dar de baja|cerrar|suspender)\b(\s+\w+){0,4}\s+(cuenta|usuario|datos)",
}
ACTION_REVIEW_PREFIX = "[REVISAR: {flags}] "

SAFE_ANSWER = "Gracias por escribirnos. Un agente de soporte va a revisar tu caso y te va a contactar."
LEAK_WINDOW = 8  # palabras seguidas del system prompt que cuentan como fuga


def normalize(text: str) -> str:
    """Minúsculas y sin tildes, para que 'Ignorá' y 'ignora' coincidan con el mismo patrón."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def deobfuscate(text: str) -> str:
    """Versión del texto sin los trucos de escritura conocidos, para pasarla por los patrones.

    1. Quita caracteres invisibles.
    2. Cambia letras cirílicas o griegas parecidas por las latinas.
    3. Cambia números o símbolos por letras, solo en palabras que también tienen letras.
    4. Une letras sueltas separadas por espacios ("i g n o r a" -> "ignora"), desde 4 letras seguidas.
    """
    text = "".join(c for c in text if unicodedata.category(c) != _INVISIBLE_CATEGORY)
    text = normalize(text).translate(_HOMOGLYPHS)

    def _leet(match: re.Match) -> str:
        word = match.group()
        has_letter = re.search(r"[a-zñ]", word)
        has_swap = re.search(r"[013457@$]", word)
        return word.translate(_LEET) if has_letter and has_swap else word

    text = re.sub(r"\S+", _leet, text)
    return re.sub(r"\b(?:\w\s){3,}\w\b", lambda m: m.group().replace(" ", ""), text)


def decoded_base64(text: str) -> list[str]:
    """Textos legibles escondidos en base64 dentro de la consulta (vacía si no hay)."""
    found = []
    for token in _BASE64_TOKEN.findall(text):
        try:
            decoded = base64.b64decode(token + "=" * (-len(token) % 4), validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        if decoded and sum(c.isprintable() or c.isspace() for c in decoded) / len(decoded) > 0.9:
            found.append(decoded)
    return found


def _match_injection(text: str) -> str | None:
    for name, pattern in INJECTION_PATTERNS.items():
        if re.search(pattern, text, flags=re.MULTILINE):
            return name
    return None


def check_input(query: str) -> tuple[str | None, str | None]:
    """Devuelve (error, injection).

    error: la consulta no se puede procesar (vacía o demasiado larga).
    injection: nombre del patrón detectado, o None si no detectó nada. Si el patrón apareció recién
    después de desofuscar, el nombre lleva "+ofuscado" o "+base64" (por ejemplo,
    "ignorar_instrucciones+ofuscado"), para saber qué truco se usó.
    """
    if not query.strip():
        return "La consulta está vacía.", None
    if len(query) > MAX_QUERY_CHARS:
        return f"La consulta supera el máximo de {MAX_QUERY_CHARS} caracteres ({len(query)}).", None
    if name := _match_injection(normalize(query)):
        return None, name
    if name := _match_injection(deobfuscate(query)):
        return None, f"{name}+ofuscado"
    for decoded in decoded_base64(query):
        if name := _match_injection(normalize(decoded)) or _match_injection(deobfuscate(decoded)):
            return None, f"{name}+base64"
    return None, None


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", normalize(text))


def _prompt_windows(system_prompt: str) -> set[tuple[str, ...]]:
    """Secuencias de LEAK_WINDOW palabras del system prompt.

    Se excluyen las líneas de ejemplos few-shot (Ticket: ... / {"answer": ...}): son respuestas
    modelo y una respuesta real puede parecerse a ellas sin que sea una fuga.
    """
    lines = [l for l in system_prompt.splitlines()
             if not l.strip().startswith(("Ticket:", '{"answer"'))]
    words = _words("\n".join(lines))
    return {tuple(words[i : i + LEAK_WINDOW]) for i in range(len(words) - LEAK_WINDOW + 1)}


def check_output(answer: str, system_prompt: str) -> list[str]:
    """Devuelve la lista de problemas encontrados en el answer (vacía = OK)."""
    text = normalize(answer)
    flags = [f"promesa:{name}" for name, pattern in PROMISE_PATTERNS.items() if re.search(pattern, text)]
    words = _words(answer)
    windows = _prompt_windows(system_prompt)
    if any(tuple(words[i : i + LEAK_WINDOW]) in windows for i in range(len(words) - LEAK_WINDOW + 1)):
        flags.append("fuga_system_prompt")
    return flags


def check_actions(actions: list[str], system_prompt: str) -> list[list[str]]:
    """Problemas de cada acción sugerida, en el mismo orden (lista vacía = esa acción está OK).

    Revisa acciones riesgosas para el agente (saltear una verificación, aprobar dinero, borrar una
    cuenta) y fragmentos del system prompt. Las promesas al cliente no aplican: las acciones no las lee
    el cliente.
    """
    windows = _prompt_windows(system_prompt)
    result = []
    for action in actions:
        text = normalize(action).strip()
        flags = [name for name, pattern in ACTION_PATTERNS.items() if re.search(pattern, text)]
        words = _words(action)
        if any(tuple(words[i : i + LEAK_WINDOW]) in windows for i in range(len(words) - LEAK_WINDOW + 1)):
            flags.append("fuga_system_prompt")
        result.append(flags)
    return result
