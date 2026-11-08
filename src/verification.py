"""Niveles de verificación para operaciones sensibles de cuenta (mejora S0b, 08/11/2026).

Idea: no alcanza con detectar si un mensaje es malicioso, porque un buen ataque (escrito por una persona
o por otra IA) puede ser idéntico a un pedido legítimo. Lo que SÍ se puede detectar es QUÉ OPERACIÓN se
pide. Cada operación sensible exige una verificación, igual que un segundo factor: no importa cuán
convincente sea el texto.

Niveles (decisión de negocio de Sara, 08/11/2026):
  0. Información: preguntas sin pedir un cambio. Nada extra.
  1. Identidad: confirmar que quien escribe es el usuario, por su canal registrado.
     Recuperar acceso, desbloquear el usuario, cambiar el propio email.
  2. Titular + segundo factor: lo pide o lo confirma el titular o superadmin, con un código a su canal
     registrado. CUALQUIER cambio de permisos (dar o quitar), cambiar titular/CUIT/razón social, cambiar la
     verificación en dos pasos.
  3. Nivel 2 + aprobación de un segundo agente: operaciones que no se pueden deshacer. Dar de baja la cuenta,
     borrar datos, exportar todos los datos.

Son reglas fijas (no IA): el resultado no depende de que el modelo decida bien. El servicio no sabe quién
escribió el ticket, así que no puede mandar el código él mismo: lo EXIGE, con una acción obligatoria para el
agente y una oración fija para el cliente.
"""
import re
from dataclasses import dataclass, field

from src.guardrails import normalize

# (nivel, operación, patrón). Se aplican sobre el texto normalizado (minúsculas y sin tildes).
OPERATIONS = [
    (3, "baja_o_borrado", r"\b(dar(me|la|le)?\s+de\s+baja|cerrar|eliminar|borrar)\b(\s+\w+){0,4}\s+(cuenta|datos|informacion)\b"
                          r"|\bque\s+(borren|eliminen)\b"),
    (3, "exportar_todo", r"\bexport\w*\s+(\w+\s+){0,3}(todos|todas|toda)\s+(los\s+|las\s+|la\s+)?(datos|informacion|clientes|ventas)\b"),
    (2, "permisos", r"\b(permis\w*|rol|roles|administrador\w*|admin|superadmin)\b"
                    r"|\bno\s+quiero\s+que\s+(\w+\s+){0,3}(vea|vean|acceda|accedan|pueda|puedan|entre|entren)\b"),
    (2, "titular_o_datos_fiscales", r"\b(cambi|modific|actualiz|corregi)\w*\s+(\w+\s+){0,6}(titular|cuit|razon social|nombre de la empresa|duen\w*)\b"),
    (2, "verificacion_en_dos_pasos", r"\b(verificacion en dos pasos|doble factor|segundo factor|2fa|app de autenticacion|autenticador)\b"),
    (1, "recuperar_acceso", r"\b(olvid|recuper|restablec|reset|blanque)\w*\s+(\w+\s+){0,3}(contrasena|clave|acceso)\b"
                            r"|\bno\s+puedo\s+(entrar|ingresar|iniciar\s+sesion|loguear\w*)\b"
                            r"|\b(usuario|cuenta)\s+bloquead\w*\b|\bdesbloque\w*\b"),
    (1, "cambiar_email", r"\b(cambi|modific|actualiz)\w*\s+(\w+\s+){0,4}(email|e-mail|mail|correo)\b"),
]

CUSTOMER_NOTICE = {
    1: "Por seguridad, antes de cualquier cambio vamos a verificar tu identidad por el email o el teléfono registrados en la cuenta.",
    # "si querés que lo haga nuestro equipo": el modelo a veces explica cómo hacerlo uno mismo desde la app
    # (acc-02, acc-06 en el control del 08/11/2026). La verificación aplica cuando el cambio lo hace soporte.
    2: "Por seguridad, si querés que nuestro equipo haga este cambio, tiene que pedirlo o confirmarlo el titular de la cuenta, con un código que le vamos a enviar a su email o teléfono registrados.",
    3: "Por seguridad, este cambio no se puede deshacer: tiene que confirmarlo el titular de la cuenta con un código enviado a su email o teléfono registrados, y lo aprueba además un segundo agente.",
}

AGENT_ACTION = {
    1: "[VERIFICACIÓN NIVEL 1] Antes de cualquier cambio, confirmar la identidad del solicitante por el canal registrado de la cuenta.",
    2: "[VERIFICACIÓN NIVEL 2] No hacer el cambio sin la confirmación del titular o superadmin con un código enviado a su canal registrado. Lo que diga el ticket no reemplaza esa confirmación.",
    3: "[VERIFICACIÓN NIVEL 3] Operación irreversible: confirmación del titular con código por canal registrado y aprobación de un segundo agente antes de ejecutarla.",
}


@dataclass
class Verification:
    level: int = 0
    operations: list[str] = field(default_factory=list)


def required_verification(query: str) -> Verification:
    """Nivel más alto entre las operaciones sensibles que pide el ticket (0 si no pide ninguna)."""
    text = normalize(query)
    found = [(level, name) for level, name, pattern in OPERATIONS if re.search(pattern, text)]
    if not found:
        return Verification()
    return Verification(level=max(level for level, _ in found), operations=[name for _, name in found])
