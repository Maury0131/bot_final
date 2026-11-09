"""El agente: modelo + instrucciones + capabilities. Sin memoria: cada mensaje empieza de cero.

Pruébalo en la terminal, sin Telegram:  uv run asistente.py
"""

import asyncio

from pydantic_ai import Agent

from capacidades import apis, consulta, despacho, documentos, registro
from config import modelo

asistente = Agent(
    modelo,
    instructions=(
        "Te llamas Killa, la asistente de gestión judicial y despacho para jueces del Tribunal Departamental "
        "de Justicia en Sucre, Bolivia. Ayudas a consultar y programar la agenda de audiencias, "
        "alertar sobre expedientes con plazos procesales por vencer, dar estadísticas del juzgado, "
        "consultar protocolos y documentos (citando la fuente), revisar el clima en Sucre para inspecciones "
        "oculares y calcular conversiones en dólares, inflación o gastos. "
        "No respondas temas ajenos a la gestión judicial o administrativa del despacho. "
        "Responde siempre en español, breve y en texto plano (sin Markdown)."
    ),
    capabilities=[apis, despacho, documentos, consulta, registro],
)


async def responder(mensaje: str) -> str:
    """Lo único que necesita el bot: un mensaje entra, una respuesta sale."""
    resultado = await asistente.run(mensaje)
    return resultado.output


async def main() -> None:
    while (mensaje := input("tú> ").strip()) not in {"", "salir"}:
        print("killa>", await responder(mensaje))


if __name__ == "__main__":
    asyncio.run(main())