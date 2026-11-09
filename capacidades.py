"""Las capabilities de tu asistente: lo que vimos en clase. Cámbialas o agrega las tuyas.

Una Capability junta tools + instrucciones en una pieza que se enchufa al agente (asistente.py).
El docstring y los tipos de cada tool son lo único que el modelo lee para decidir cuándo usarla.
"""

import csv
import sqlite3
from datetime import date
from functools import cache
from typing import Literal

import chromadb
import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pydantic_ai.capabilities import Capability
from pypdf import PdfReader

from config import DATOS, settings

# ═══ S1 · Tools de lectura con APIs públicas ══════════════════════════════════════════
apis = Capability(
    id="apis_publicas",
    description="Consulta indicadores económicos (Banco Mundial), tipo de cambio (ExchangeRate) y clima (Open Meteo)",
    instructions="Usa las APIs para obtener datos reales. Si una API no tiene el dato, dilo, y no inventes datos",
)


@apis.tool_plain
def inflacion(pais: str, anio: int) -> dict:
    """Inflación anual de un país en un año, según el Banco Mundial.

    Args:
        pais: código ISO del país, por ejemplo 'BO' para Bolivia o 'PE' para Perú
        anio: año, por ejemplo 2024
    """
    url = f"https://api.worldbank.org/v2/country/{pais}/indicator/FP.CPI.TOTL.ZG"
    respuesta = httpx.get(url, params={"date": anio, "format": "json"}, timeout=10)
    respuesta.raise_for_status()
    datos = respuesta.json()

    if len(datos) < 2 or not datos[1] or datos[1][0]["value"] is None:
        return {"error": f"No hay datos de inflación disponibles para {pais} en {anio}"}

    registro = datos[1][0]
    return {
        "pais": registro["country"]["value"],
        "anio": registro["date"],
        "inflacion_pct": round(registro["value"], 2),
        "fuente": "Banco Mundial",
    }


@apis.tool_plain
def clima_cochabamba() -> dict:
    """Temperatura actual de Cochabamba, según Open Meteo."""
    params = {"latitude": -17.38, "longitude": -66.15, "current": "temperature_2m"}
    respuesta = httpx.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=10)
    respuesta.raise_for_status()
    return {"temperatura": respuesta.json()["current"]["temperature_2m"], "fuente": "Open Meteo"}


@apis.tool_plain
def clima_sucre() -> dict:
    """Temperatura actual, lluvia y viento en Sucre, Bolivia, para planificar inspecciones judiciales (Open Meteo)."""
    params = {
        "latitude": -19.0333,
        "longitude": -65.2627,
        "current": ["temperature_2m", "precipitation", "wind_speed_10m"],
        "timezone": "America/La_Paz",
    }
    respuesta = httpx.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=10)
    respuesta.raise_for_status()
    actual = respuesta.json()["current"]
    return {
        "ciudad": "Sucre",
        "temperatura_c": actual["temperature_2m"],
        "precipitacion_mm": actual["precipitation"],
        "viento_kmh": actual["wind_speed_10m"],
        "fuente": "Open Meteo",
    }


@apis.tool_plain
def tipo_cambio_usd() -> dict:
    """Tipo de cambio oficial de dólar (USD) a bolivianos (BOB) según ExchangeRate-API para liquidaciones o fianzas."""
    respuesta = httpx.get("https://open.er-api.com/v6/latest/USD", timeout=10)
    respuesta.raise_for_status()
    tasa = respuesta.json()["rates"]["BOB"]
    return {"moneda_base": "USD", "moneda_destino": "BOB", "tipo_cambio": tasa, "fuente": "ExchangeRate-API"}


# ═══ S1 · Mi contador: gastos en SQLite (se mantiene para compatibilidad con pytest) ═══
Categoria = Literal["transporte", "alimentacion", "educacion", "ocio", "salud"]
CSV = DATOS / "gastos.csv"
DB = DATOS / "gastos.db"


def crear_base() -> None:
    """Crea gastos.db desde cero con los gastos del CSV."""
    DB.unlink(missing_ok=True)
    with CSV.open(encoding="utf-8") as archivo:
        filas = [(g["fecha"], g["categoria"], g["descripcion"], float(g["monto"])) for g in csv.DictReader(archivo)]
    conexion = sqlite3.connect(DB)
    with conexion:
        conexion.execute(
            "CREATE TABLE gastos (fecha TEXT, categoria TEXT, descripcion TEXT, monto REAL CHECK (monto > 0))"
        )
        conexion.executemany("INSERT INTO gastos VALUES (?, ?, ?, ?)", filas)
    conexion.close()


def hoy() -> str:
    """Instrucción dinámica: el modelo no sabe qué día es; se lo decimos en cada mensaje."""
    return f"Hoy es {date.today().isoformat()}."


consulta = Capability(
    id="consulta_gastos",
    description="Consulta gastos personales almacenados en una base de datos SQLite",
    instructions="Usa la base de datos para responder preguntas sobre los gastos. No inventes datos.",
)
registro = Capability(
    id="registro_gastos",
    description="Registra nuevos gastos en la base de datos SQLite",
    instructions=["Agrega nuevos gastos a la base de datos. Asegúrate de que los datos sean válidos.", hoy],
)


@consulta.tool_plain
def consultar_gastos(categoria: Categoria) -> dict:
    """Gastos de una categoría y su total en bolivianos."""
    if not DB.exists():
        crear_base()
    conexion = sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True)
    conexion.row_factory = sqlite3.Row
    filas = conexion.execute(
        "SELECT fecha, descripcion, monto FROM gastos WHERE categoria = ?",
        (categoria,),
    ).fetchall()
    conexion.close()
    return {"categoria": categoria, "total_bs": sum(f["monto"] for f in filas), "gastos": [dict(f) for f in filas]}


@registro.tool_plain
def registrar_gasto(fecha: str, categoria: Categoria, descripcion: str, monto: float) -> str:
    """Anota un gasto nuevo. La fecha va como AAAA-MM-DD y el monto en bolivianos."""
    if not DB.exists():
        crear_base()
    conexion = sqlite3.connect(DB)
    try:
        with conexion:
            conexion.execute("INSERT INTO gastos VALUES (?, ?, ?, ?)", (fecha, categoria, descripcion, monto))
    except sqlite3.IntegrityError:
        return f"No se guardó: el monto debe ser mayor que 0 (recibí {monto})."
    finally:
        conexion.close()
    return f"Gasto registrado: {descripcion}, {monto} Bs en {categoria} el {fecha}."


# ═══ S1-B · Despacho Judicial: Audiencias, Plazos y Estadísticas en SQLite ═════════════
Materia = Literal["Familiar", "Civil", "Laboral"]
CSV_CAUSAS = DATOS / "causas.csv"
DB_CAUSAS = DATOS / "causas.db"


def crear_base_judicial() -> None:
    """Crea causas.db desde causas.csv la primera vez."""
    DB_CAUSAS.unlink(missing_ok=True)
    with CSV_CAUSAS.open(encoding="utf-8") as archivo:
        filas = [
            (
                c["expediente"],
                c["materia"],
                c["partes"],
                c["tipo_acto"],
                c["fecha_audiencia"],
                c["hora"],
                c["estado"],
                c["fecha_limite_plazo"],
            )
            for c in csv.DictReader(archivo)
        ]
    conexion = sqlite3.connect(DB_CAUSAS)
    with conexion:
        conexion.execute(
            """CREATE TABLE causas (
                expediente TEXT, materia TEXT, partes TEXT, tipo_acto TEXT,
                fecha_audiencia TEXT, hora TEXT, estado TEXT, fecha_limite_plazo TEXT
            )"""
        )
        conexion.executemany("INSERT INTO causas VALUES (?, ?, ?, ?, ?, ?, ?, ?)", filas)
    conexion.close()


despacho = Capability(
    id="despacho_judicial",
    description="Consulta y programa audiencias judiciales, revisa plazos por vencer y estadísticas del juzgado",
    instructions=[
        "Usa las herramientas de despacho para responder sobre audiencias, expedientes, estadísticas y plazos. "
        "No inventes expedientes ni nombres de partes.",
        hoy,
    ],
)


@despacho.tool_plain
def consultar_audiencias(fecha: str | None = None, materia: Materia | None = None) -> dict:
    """Lista las audiencias y causas filtrando opcionalmente por fecha (AAAA-MM-DD) o materia (Familiar, Civil, Laboral)."""
    if not DB_CAUSAS.exists():
        crear_base_judicial()
    conexion = sqlite3.connect(DB_CAUSAS.resolve().as_uri() + "?mode=ro", uri=True)
    conexion.row_factory = sqlite3.Row
    sql = "SELECT * FROM causas WHERE 1=1"
    params: list[str] = []
    if fecha:
        sql += " AND fecha_audiencia = ?"
        params.append(fecha)
    if materia:
        sql += " AND materia = ?"
        params.append(materia)
    sql += " ORDER BY fecha_audiencia, hora"
    filas = conexion.execute(sql, params).fetchall()
    conexion.close()
    return {"total": len(filas), "audiencias": [dict(f) for f in filas]}


@despacho.tool_plain
def procesos_por_vencer(fecha_hasta: str) -> dict:
    """Devuelve los expedientes cuyo plazo procesal vence en o antes de la fecha indicada (AAAA-MM-DD), ordenados por urgencia."""
    if not DB_CAUSAS.exists():
        crear_base_judicial()
    conexion = sqlite3.connect(DB_CAUSAS.resolve().as_uri() + "?mode=ro", uri=True)
    conexion.row_factory = sqlite3.Row
    filas = conexion.execute(
        "SELECT expediente, materia, partes, tipo_acto, estado, fecha_limite_plazo "
        "FROM causas WHERE fecha_limite_plazo <= ? ORDER BY fecha_limite_plazo ASC",
        (fecha_hasta,),
    ).fetchall()
    conexion.close()
    return {"urgentes_encontrados": len(filas), "expedientes": [dict(f) for f in filas]}


@despacho.tool_plain
def estadisticas_juzgado() -> dict:
    """Calcula el total de causas por materia y por estado procesal (Pendiente vs En Despacho)."""
    if not DB_CAUSAS.exists():
        crear_base_judicial()
    conexion = sqlite3.connect(DB_CAUSAS.resolve().as_uri() + "?mode=ro", uri=True)
    por_materia = dict(conexion.execute("SELECT materia, COUNT(*) FROM causas GROUP BY materia").fetchall())
    por_estado = dict(conexion.execute("SELECT estado, COUNT(*) FROM causas GROUP BY estado").fetchall())
    conexion.close()
    return {"total_causas": sum(por_materia.values()), "por_materia": por_materia, "por_estado": por_estado}


@despacho.tool_plain
def programar_audiencia(
    expediente: str,
    materia: Materia,
    partes: str,
    tipo_acto: str,
    fecha_audiencia: str,
    hora: str,
    fecha_limite_plazo: str,
) -> str:
    """Programa una nueva audiencia en el juzgado verificando que no haya choque de fecha y hora."""
    if not DB_CAUSAS.exists():
        crear_base_judicial()
    conexion = sqlite3.connect(DB_CAUSAS)
    try:
        choque = conexion.execute(
            "SELECT expediente, partes FROM causas WHERE fecha_audiencia = ? AND hora = ?",
            (fecha_audiencia, hora),
        ).fetchone()
        if choque:
            return f"No se puede agendar: ya existe audiencia el {fecha_audiencia} a las {hora} ({choque[0]} - {choque[1]})."
        with conexion:
            conexion.execute(
                "INSERT INTO causas VALUES (?, ?, ?, ?, ?, ?, 'Pendiente', ?)",
                (expediente, materia, partes, tipo_acto, fecha_audiencia, hora, fecha_limite_plazo),
            )
    finally:
        conexion.close()
    return f"Audiencia programada en Killa: {expediente} ({partes}) el {fecha_audiencia} a las {hora}."


# ═══ S2 · RAG: buscar en mis documentos (PDFs) y citar la fuente ══════════════════════
DOCUMENTOS = DATOS / "documentos"


def embeddings_de(textos: list[str]) -> list[list[float]]:
    """Convierte cada texto en un vector: textos de significado parecido dan vectores cercanos."""
    cliente = OpenAI(
        api_key=settings.gemini_api_key_local.get_secret_value(),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    respuesta = cliente.embeddings.create(model="gemini-embedding-001", input=textos)
    return [item.embedding for item in respuesta.data]


@cache
def indice():
    coleccion = chromadb.EphemeralClient().get_or_create_collection(
        "documentos", configuration={"hnsw": {"space": "cosine"}}
    )
    for pdf in sorted(DOCUMENTOS.glob("*.pdf")):
        texto = "\n".join(pagina.extract_text() for pagina in PdfReader(pdf).pages)
        trozos = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_text(texto)
        coleccion.upsert(
            ids=[f"{pdf.stem}-{i}" for i in range(len(trozos))],
            documents=trozos,
            embeddings=embeddings_de(trozos),
        )
    return coleccion


def buscar(pregunta: str, k: int = 3) -> list[dict]:
    """Los k trozos más parecidos a la pregunta, con su fuente y su similitud (1 = idéntico)."""
    r = indice().query(query_embeddings=embeddings_de([pregunta]), n_results=k)
    return [
        {"fuente": fuente, "similitud": round(1 - distancia, 2), "texto": texto}
        for fuente, distancia, texto in zip(r["ids"][0], r["distances"][0], r["documents"][0])
    ]


documentos = Capability(
    id="mis_documentos",
    description="Documentos del despacho, protocolos judiciales, plazos procesales, contratos y normativa",
    instructions=(
        "Para preguntas sobre plazos legales, protocolos de audiencias, contratos, pagos o normativa "
        "usa buscar_documentos y cita la fuente entre corchetes, por ejemplo [protocolo_audiencias-1]. "
        "Si no hay información suficiente para responder, di que no lo encuentras, no lo inventes."
    ),
)


@documentos.tool_plain
def buscar_documentos(pregunta: str) -> list[dict]:
    """Busca en los documentos los fragmentos que responden a la pregunta."""
    resultados = buscar(pregunta, k=3)
    for r in resultados:
        print(f"  📄 {r['fuente']} (similitud {r['similitud']})", flush=True)
    return resultados