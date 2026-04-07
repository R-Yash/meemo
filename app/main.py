import os
import time

import psycopg2
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.db import lifespan, get_pg_pool, get_neo4j, get_pinecone
from app.models import HealthResponse
from app.routers import ingest, query, documents

app = FastAPI(
    title="Meemo API",
    description=(
        "Personal knowledge base with hybrid RAG retrieval. "
        "Ingest notes, URLs, YouTube videos and files; query them with natural language."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.1f}"
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": "An unexpected error occurred.", "type": type(exc).__name__})

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(documents.router)

@app.get("/health", response_model=HealthResponse, tags=["health"])
def health_check():
    pg_ok = neo4j_ok = pc_ok = False

    try:
        pool = get_pg_pool()
        conn = pool.getconn()
        cur  = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        pool.putconn(conn)
        pg_ok = True
    except Exception as exc:
        pass

    try:
        driver = get_neo4j()
        with driver.session() as session:
            session.run("RETURN 1")
        neo4j_ok = True
    except Exception as exc:
        pass

    try:
        pc     = get_pinecone()
        _index = pc.Index(os.environ["PINECONE_INDEX"])
        _index.describe_index_stats()
        pc_ok = True
    except Exception as exc:
        pass

    all_ok  = pg_ok and neo4j_ok and pc_ok
    payload = HealthResponse(
        status="ok" if all_ok else "degraded",
        postgres=pg_ok,
        pinecone=pc_ok,
        neo4j=neo4j_ok,
    )

    return JSONResponse(
        status_code=200 if all_ok else 503,
        content=payload.model_dump(),
    )

@app.get("/", include_in_schema=False)
def root():
    return {"message": "Meemo API is running. Visit /docs for the interactive API reference."}