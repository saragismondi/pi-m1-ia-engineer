"""Carga los prompts versionados desde prompts/*.yaml.

Los prompts viven fuera del código: para probar una versión nueva se agrega un YAML, no se toca Python.
"""
from dataclasses import dataclass
from pathlib import Path

import yaml

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@dataclass
class Prompt:
    version: str
    technique: str
    system: str
    user_template: str
    # Marcado del ticket ("spotlighting" por datamarking, C1, 08/11/2026): si el YAML define `datamark`, ese
    # carácter reemplaza los espacios del texto del cliente. Así el modelo distingue a simple vista qué texto
    # es un dato del cliente, aunque el atacante escriba algo que parezca una instrucción. Opcional.
    datamark: str | None = None

    def render_user(self, query: str) -> str:
        if self.datamark:
            query = " ".join(query.split()).replace(" ", self.datamark)
        # replace() y no format(): el ticket puede traer llaves { } que romperían format().
        return self.user_template.replace("{query}", query)


def available_versions() -> list[str]:
    return sorted(path.stem.split("_")[0] for path in PROMPTS_DIR.glob("v*.yaml"))


def load_prompt(version: str) -> Prompt:
    """Busca prompts/<version>_*.yaml (por ejemplo, v1 -> v1_zero_shot.yaml)."""
    matches = sorted(PROMPTS_DIR.glob(f"{version}_*.yaml"))
    if not matches:
        raise ValueError(f"No existe el prompt {version!r}. Disponibles: {', '.join(available_versions())}")
    data = yaml.safe_load(matches[0].read_text(encoding="utf-8"))
    return Prompt(
        version=data["version"],
        technique=data["technique"],
        system=data["system"],
        user_template=data["user_template"],
        datamark=data.get("datamark"),
    )
