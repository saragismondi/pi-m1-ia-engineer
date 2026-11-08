"""Runner de evals: corre el dataset contra cada combinación proveedor x prompt y reporta el score.

Uso:
    python evals/runner.py                                  # todos los proveedores y prompts
    python evals/runner.py --provider groq --prompt v1 v2   # una parte
    python evals/runner.py --limit 4                        # prueba corta con 4 casos

Score = exact match: porcentaje de casos donde la categoría predicha es igual a la esperada.
Un caso que falla (error de API o contrato inválido tras los reintentos) cuenta como incorrecto.
output_flagged = respuestas donde el guardrail de salida detectó una promesa o una fuga del prompt.

Guarda el detalle de cada caso en evals/results/<fecha>_<proveedor>_<prompt>.jsonl
y escribe la tabla resumen en evals/results/<fecha>_summary.md.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # permite correrlo como `python evals/runner.py`

from dotenv import load_dotenv  # noqa: E402

from src.metrics import metrics_path  # noqa: E402
from src.prompts import available_versions  # noqa: E402
from src.providers import PROVIDERS  # noqa: E402
from src.schemas import CATEGORIES  # noqa: E402
from src.service import TriageError, triage  # noqa: E402

DATASET = ROOT / "evals" / "dataset.jsonl"
RESULTS_DIR = ROOT / "evals" / "results"


def load_dataset(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def count_metric_lines() -> int:
    if not metrics_path().exists():
        return 0
    return len(metrics_path().read_text(encoding="utf-8").splitlines())


def read_metric_lines_since(start: int) -> list[dict]:
    """Las líneas que agregó esta corrida a data/metrics.jsonl (para sumar costo y latencia)."""
    lines = metrics_path().read_text(encoding="utf-8").splitlines()[start:]
    return [json.loads(line) for line in lines if line.strip()]


def run_one(provider: str, prompt: str, cases: list[dict], sleep_s: float) -> list[dict]:
    results = []
    for i, case in enumerate(cases, 1):
        answer, output_flags, actions, action_flags = None, [], [], []
        try:
            # Sin caché: cada corrida tiene que medir llamadas reales.
            result = triage(case["query"], provider, prompt, use_cache=False)
            predicted, error = result.response.category, None
            answer = result.original_answer or result.response.answer
            output_flags = result.output_flags
            actions, action_flags = result.response.actions, result.action_flags
        except TriageError as e:
            predicted, error = None, str(e)
        results.append({
            "id": case["id"],
            "type": case["type"],
            "expected": case["expected_category"],
            "predicted": predicted,
            "correct": predicted == case["expected_category"],
            "error": error,
            "answer": answer,              # answer del modelo (antes de un eventual reemplazo)
            "output_flags": output_flags,  # lo que detectó el guardrail de salida
            "actions": actions,            # acciones finales (las riesgosas llevan [REVISAR])
            "action_flags": action_flags,  # lo que detectó el guardrail de acciones
        })
        mark = "ok " if results[-1]["correct"] else "MAL"
        print(f"  [{i:>2}/{len(cases)}] {mark} {case['id']:<7} esperado={case['expected_category']:<9} predicho={predicted}")
        if sleep_s:
            time.sleep(sleep_s)  # para no chocar con el rate limit del plan gratis
    return results


def pct(correct: int, total: int) -> str:
    return f"{100 * correct / total:.0f}% ({correct}/{total})" if total else "-"


def summarize(provider: str, prompt: str, results: list[dict], metrics: list[dict]) -> dict:
    by_cat = defaultdict(lambda: [0, 0])  # categoría -> [correctos, total]
    for r in results:
        by_cat[r["expected"]][0] += r["correct"]
        by_cat[r["expected"]][1] += 1
    clear = [r for r in results if r["type"] == "claro"]
    costs = [m["cost_usd"] for m in metrics if m["cost_usd"] is not None]
    latencies = [m["latency_s"] for m in metrics]
    return {
        "provider": provider,
        "prompt": prompt,
        "n": len(results),
        "global": pct(sum(r["correct"] for r in results), len(results)),
        "clear_only": pct(sum(r["correct"] for r in clear), len(clear)),
        **{cat: pct(*by_cat[cat]) for cat in CATEGORIES},
        "failed": sum(r["error"] is not None for r in results),
        "output_flagged": sum(bool(r["output_flags"]) for r in results),
        "requests": len(metrics),
        "retries": sum(m["attempt"] > 1 for m in metrics),
        "avg_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "total_tokens": sum(m["total_tokens"] for m in metrics),
        "cost_usd": round(sum(costs), 6) if len(costs) == len(metrics) else None,
    }


def markdown_table(summaries: list[dict]) -> str:
    cols = ["prompt", "provider", "n", "global", "clear_only", *CATEGORIES,
            "failed", "output_flagged", "retries", "avg_latency_s", "total_tokens", "cost_usd"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for s in summaries:
        lines.append("| " + " | ".join(str(s[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(description="Evalúa el triage contra evals/dataset.jsonl")
    parser.add_argument("--provider", nargs="+", choices=PROVIDERS, default=list(PROVIDERS))
    parser.add_argument("--prompt", nargs="+", default=available_versions())
    parser.add_argument("--limit", type=int, default=None, help="Usar solo los primeros N casos.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Segundos de espera entre llamadas.")
    args = parser.parse_args()

    cases = load_dataset(DATASET, args.limit)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []

    for provider in args.provider:
        for prompt in args.prompt:
            print(f"\n== {provider} / {prompt} ({len(cases)} casos)")
            start = count_metric_lines()
            results = run_one(provider, prompt, cases, args.sleep)
            summary = summarize(provider, prompt, results, read_metric_lines_since(start))
            summaries.append(summary)
            detail_path = RESULTS_DIR / f"{stamp}_{provider}_{prompt}.jsonl"
            detail_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")

    table = markdown_table(summaries)
    (RESULTS_DIR / f"{stamp}_summary.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
