"""Une todo el flujo de una consulta:

consulta -> guardrail de entrada -> caché -> prompt -> proveedor LLM
         -> parseo + validación Pydantic (con reintentos) -> guardrail de salida -> métricas
"""
from dataclasses import dataclass, field

from pydantic import ValidationError

import os
import time

from src import cache
from src.guardrails import ACTION_REVIEW_PREFIX, SAFE_ANSWER, check_actions, check_input, check_output
from src.metrics import log_event, log_request
from src.moderation import moderate
from src.prompts import load_prompt
from src.providers import API_ERRORS, get_provider
from src.schemas import TriageResponse, extract_json

# Moderador LLM (src/moderation.py): off | output | both. "both" modera entrada y salida.
MODERATION_MODES = ("off", "output", "both")

# Reintentos ante una respuesta que no respeta el contrato JSON.
# (Los errores de red y rate limits los reintenta el SDK; ver providers.py.)
MAX_ATTEMPTS = 3


class TriageError(Exception):
    """Error esperado y explicable: la CLI lo muestra como JSON de error, sin traceback."""


@dataclass
class TriageResult:
    """La respuesta del contrato más lo que pasó por el camino (para la CLI y los evals)."""
    response: TriageResponse
    input_flag: str | None = None                          # patrón de injection detectado (no se llamó al LLM)
    output_flags: list[str] = field(default_factory=list)  # problemas del answer original (se reemplazó)
    original_answer: str | None = None                     # answer del modelo antes del reemplazo
    action_flags: list[str] = field(default_factory=list)  # problemas de las acciones (se marcaron con [REVISAR])
    cached: bool = False


def parse_response(text: str) -> TriageResponse:
    """Extrae el objeto JSON del texto del modelo y lo valida contra el contrato.

    Algunos modelos envuelven el JSON en ```json ... ``` o agregan texto: tomamos
    desde la primera '{' hasta la última '}'.
    """
    return TriageResponse.model_validate(extract_json(text))


def describe_error(error: Exception) -> str:
    """Mensaje corto para el modelo: qué campo falló y por qué."""
    if isinstance(error, ValidationError):
        return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
    return str(error)


def blocked_response(reason: str) -> TriageResponse:
    """Respuesta segura cuando el guardrail de entrada bloquea: respeta el contrato y deriva a un humano."""
    return TriageResponse(
        answer="No pudimos procesar tu consulta automáticamente. Un agente de soporte la va a revisar.",
        confidence=0.0,
        category="other",
        actions=[f"Revisar el ticket manualmente: el guardrail de entrada detectó un posible intento de manipulación ({reason})."],
    )


def call_llm(provider, prompt, query: str) -> TriageResponse:
    """Llama al modelo y valida el contrato; reintenta con el error como feedback."""
    system = prompt.system
    user = prompt.render_user(query)
    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if last_error:
            # La corrección va en el system prompt: es una instrucción nuestra, no un dato del cliente.
            system = prompt.system + (
                f"\n\nTu respuesta anterior no respetó el contrato ({last_error}). "
                "Respondé solo con el objeto JSON corregido."
            )
        try:
            result = provider.complete(system, user)
        except API_ERRORS as e:
            raise TriageError(f"Error del proveedor {provider.name}: {type(e).__name__}: {e}") from e

        try:
            response = parse_response(result.text)
        except (ValueError, ValidationError) as e:  # json.JSONDecodeError es un ValueError
            last_error = describe_error(e)
            if result.stop_reason in ("max_tokens", "length"):  # Anthropic / OpenAI-Groq
                last_error += " (la respuesta se cortó por MAX_TOKENS)"
            log_request(result, prompt.version, attempt, valid=False, category=None)
            continue

        log_request(result, prompt.version, attempt, valid=True, category=response.category)
        return response

    raise TriageError(f"Se agotaron los {MAX_ATTEMPTS} intentos sin una respuesta válida. Último error: {last_error}")


def triage(query: str, provider_name: str, prompt_version: str = "v1",
           use_cache: bool = True, input_guardrail: bool = True, moderation: str | None = None) -> TriageResult:
    """input_guardrail=False apaga las dos capas de entrada (patrones y moderador): solo existe
    para medir su efecto en los evals adversariales.
    moderation: off | output | both (por defecto, la variable MODERATION del .env, o "both").
    """
    start = time.perf_counter()  # latencia del servicio, para las ejecuciones que no llaman al LLM
    moderation = moderation or os.getenv("MODERATION", "both")
    if moderation not in MODERATION_MODES:
        raise TriageError(f"MODERATION inválido: {moderation!r}. Opciones: {', '.join(MODERATION_MODES)}")
    error, injection = check_input(query)
    if error:
        raise TriageError(error)
    if injection and input_guardrail:
        # Todavía no se creó el proveedor (no hace falta para bloquear): el modelo queda sin dato.
        log_event("bloqueado_patron", provider_name, None, prompt_version, "other", time.perf_counter() - start)
        return TriageResult(response=blocked_response(injection), input_flag=injection)

    try:
        provider = get_provider(provider_name)
        # El moderador puede ser otro proveedor (por defecto Haiku): así no consume el cupo diario
        # del plan gratis de Groq, que se agotó cuando el moderador corría también en Groq.
        moderator = get_provider(os.getenv("MODERATOR_PROVIDER", "anthropic")) if moderation != "off" else None
        prompt = load_prompt(prompt_version)
    except ValueError as e:  # proveedor o prompt inexistente, o falta una clave
        raise TriageError(str(e)) from e

    key = cache.make_key(provider.name, provider.model, prompt.version, query)
    if use_cache and (hit := cache.get(key)):
        response = TriageResponse.model_validate(hit)
        log_event("cache", provider.name, provider.model, prompt.version, response.category,
                  time.perf_counter() - start)
        return TriageResult(response=response, cached=True)

    # Capa 2 de entrada: el moderador LLM, para ataques que los patrones no reconocen.
    if input_guardrail and moderation == "both":
        verdict = moderate(query, "entrada", moderator)
        if verdict.flagged:
            reason = "moderador:" + ",".join(verdict.reasons)
            # La llamada al moderador ya quedó registrada (con sus tokens); esta línea registra
            # que la ejecución terminó bloqueada, sin llamar al modelo que responde.
            log_event("bloqueado_moderador", provider.name, provider.model, prompt.version, "other",
                      time.perf_counter() - start)
            return TriageResult(response=blocked_response(reason), input_flag=reason)

    response = call_llm(provider, prompt, query)

    # Acciones para el agente (A2): las riesgosas no se borran, se marcan con [REVISAR] para que el
    # agente las mire antes de seguirlas. Borrarlas le escondería información a quien decide.
    per_action = check_actions(response.actions, prompt.system)
    action_flags = sorted({f for flags_i in per_action for f in flags_i})
    if action_flags:
        response = response.model_copy(update={"actions": [
            ACTION_REVIEW_PREFIX.format(flags=", ".join(f)) + a if f else a
            for a, f in zip(response.actions, per_action)
        ]})

    result = TriageResult(response=response, action_flags=action_flags)
    flags = check_output(response.answer, prompt.system)
    # Capa 2 de salida: si los patrones no vieron nada, el moderador busca paráfrasis.
    if not flags and moderation in ("output", "both"):
        verdict = moderate(response.answer, "salida", moderator)
        flags = ["moderador:" + r for r in verdict.reasons]
    if flags:
        result.output_flags = flags
        result.original_answer = response.answer
        result.response = response.model_copy(update={
            "answer": SAFE_ANSWER,
            "actions": response.actions + [f"Redactar la respuesta al cliente: el guardrail de salida reemplazó la original ({', '.join(flags)})."],
        })

    if use_cache:
        cache.put(key, result.response.model_dump())
    return result
