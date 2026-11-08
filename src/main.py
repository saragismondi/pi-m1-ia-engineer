"""CLI de triage de soporte Nubbix.

Uso:
    python -m src.main "me cobraron dos veces la suscripción"
    python -m src.main "no puedo entrar" --provider anthropic --prompt v1
    python -m src.main "no puedo entrar" --no-cache

Salida: JSON con {answer, confidence, category, actions} en stdout (código de salida 0).
Si algo falla: JSON {"error": "..."} en stderr (código de salida 1).
"""
import argparse
import json
import os
import sys

from dotenv import load_dotenv

from src.providers import PROVIDERS
from src.service import MODERATION_MODES, TriageError, triage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clasifica una consulta de soporte de Nubbix.")
    parser.add_argument("query", nargs="+", help="Texto libre de la consulta del cliente.")
    parser.add_argument("--provider", choices=PROVIDERS, default=None,
                        help="Proveedor LLM (por defecto, DEFAULT_PROVIDER del .env).")
    parser.add_argument("--prompt", default="v1", help="Versión del prompt (archivo prompts/<versión>_*.yaml).")
    parser.add_argument("--no-cache", action="store_true", help="No usar la caché: siempre llama al LLM.")
    parser.add_argument("--moderation", choices=MODERATION_MODES, default=None,
                        help="Moderador LLM: off, output o both (por defecto, MODERATION del .env, o both).")
    return parser


def main() -> int:
    # La consola de Windows no usa UTF-8 por defecto: sin esto, los acentos pueden fallar.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    load_dotenv()

    args = build_parser().parse_args()
    query = " ".join(args.query)
    provider = args.provider or os.getenv("DEFAULT_PROVIDER", "groq")

    try:
        result = triage(query, provider, args.prompt, use_cache=not args.no_cache,
                        moderation=args.moderation)
    except TriageError as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False), file=sys.stderr)
        return 1

    # Avisos para el operador en stderr: stdout queda solo con el JSON del contrato.
    if result.input_flag:
        print(f"[guardrail de entrada] bloqueado: {result.input_flag} (no se generó respuesta)", file=sys.stderr)
    if result.output_flags:
        print(f"[guardrail de salida] answer reemplazado: {', '.join(result.output_flags)}", file=sys.stderr)
    if result.action_flags:
        print(f"[guardrail de acciones] acciones marcadas con [REVISAR]: {', '.join(result.action_flags)}", file=sys.stderr)
    if result.cached:
        print("[caché] respuesta guardada, no se llamó al LLM", file=sys.stderr)

    print(json.dumps(result.response.model_dump(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
