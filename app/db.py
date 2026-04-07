import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import psycopg2
from psycopg2 import pool as pg_pool
from pinecone import Pinecone
from neo4j import GraphDatabase, Driver
from fastapi import FastAPI

_pg_pool: pg_pool.ThreadedConnectionPool | None = None
_neo4j: Driver | None = None
_pinecone: Pinecone | None = None

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _pg_pool, _neo4j, _pinecone

    _pg_pool = pg_pool.ThreadedConnectionPool(
        minconn=2,
        maxconn=10,
        dsn=os.environ["POSTGRES_DSN"],
    )

    _neo4j = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
        max_connection_pool_size=10,
    )
    _neo4j.verify_connectivity()

    _pinecone = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    yield

    if _pg_pool:
        _pg_pool.closeall()
    if _neo4j:
        _neo4j.close()

def get_pg_pool() -> pg_pool.ThreadedConnectionPool:
    assert _pg_pool is not None, "Postgres pool not initialised"
    return _pg_pool

def get_neo4j() -> Driver:
    assert _neo4j is not None, "Neo4j driver not initialised"
    return _neo4j

def get_pinecone() -> Pinecone:
    assert _pinecone is not None, "Pinecone client not initialised"
    return _pinecone

class PgConn:
    def __init__(self):
        self._conn = None

    def __enter__(self) -> psycopg2.extensions.connection:
        self._conn = get_pg_pool().getconn()
        return self._conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self._conn.rollback()
        else:
            self._conn.commit()
        get_pg_pool().putconn(self._conn)
        return False