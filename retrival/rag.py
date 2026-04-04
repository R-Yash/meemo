import os
import json
import asyncio
import hashlib
from dataclasses import dataclass, field
from dotenv import load_dotenv
 
import psycopg2
from google import genai
from google.genai import types
from pinecone import Pinecone
from neo4j import GraphDatabase

load_dotenv()
 
PINECONE_KEY = os.getenv("PINECONE_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
POSTGRES_DSN = os.getenv("POSTGRES_DSN")
 
client = genai.Client()
 
SEMANTIC_TOP_K = 10   
GRAPH_HOP_DEPTH = 1   
GRAPH_CHUNK_K  = 10   
 
@dataclass
class QueryAnalysis:
    original: str
    rewritten: str
    entities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    text: str
    score: float
    source: str        
    title: str = ""
    source_type: str = ""
 
@dataclass
class GraphContext:
    triples:        list[tuple[str, str, str]]  
    entity_doc_ids: list[str]

def _entity_id(name: str) -> str:
    return hashlib.sha256(name.strip().lower().encode()).hexdigest()[:16]

async def analyze_query(query: str) -> QueryAnalysis:
    prompt = (
        "Analyze the following search query for a personal knowledge base.\n"
        "Return ONLY a JSON object with these exact keys:\n"
        "  'rewritten' — a clearer, context-expanded version of the query optimised for semantic search\n"
        "  'entities'  — list of named entities (people, orgs, places, concepts, events) mentioned\n"
        "  'tags'      — 2–4 lowercase topic tags this query most likely relates to\n\n"
        f"Query: {query}"
    )
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    try:
        data = json.loads(response.text)
        return QueryAnalysis(
            original=query,
            rewritten=data.get("rewritten", query),
            entities=[e.strip() for e in data.get("entities", [])],
            tags=[t.lower().strip() for t in data.get("tags", [])],
        )
    except json.JSONDecodeError:
        return QueryAnalysis(original=query, rewritten=query)

async def semantic_search(analysis: QueryAnalysis, top_k: int = SEMANTIC_TOP_K) -> list[RetrievedChunk]:
    embed_result = await client.aio.models.embed_content(
        model="gemini-embedding-2-preview",
        contents=analysis.rewritten,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=768)
    )
    query_vector = embed_result.embeddings[0].values
 
    pc = Pinecone(api_key=PINECONE_KEY)
    index = pc.Index(PINECONE_INDEX)
 
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )
 
    return [
        RetrievedChunk(
            chunk_id=match.id,
            document_id=match.metadata.get("document_id", ""),
            text=match.metadata.get("text", ""),
            score=match.score,
            source="semantic",
            title=match.metadata.get("title", ""),
            source_type=match.metadata.get("source_type", ""),
        )
        for match in response.matches
        if match.metadata
    ]

def graph_search(
    analysis: QueryAnalysis,
    driver,
    hop_depth: int = GRAPH_HOP_DEPTH,
) -> tuple[list[RetrievedChunk], GraphContext]:
    if not analysis.entities:
        return [], GraphContext(triples=[], entity_doc_ids=[])
 
    seed_ids = [_entity_id(e) for e in analysis.entities]
 
    with driver.session() as session:
        if hop_depth == 1:
            rel_rows = session.run(
                """
                MATCH (a:Entity)-[r]->(b:Entity)
                WHERE a.id IN $ids OR b.id IN $ids
                RETURN a.name AS source, type(r) AS relation, b.name AS target
                LIMIT 60
                """,
                ids=seed_ids,
            )
        else:
            rel_rows = session.run(
                """
                MATCH (seed:Entity)-[*1..2]-(neighbor:Entity)
                WHERE seed.id IN $ids
                WITH collect(DISTINCT neighbor.id) + $ids AS all_ids
                MATCH (a:Entity)-[r]->(b:Entity)
                WHERE a.id IN all_ids AND b.id IN all_ids
                RETURN a.name AS source, type(r) AS relation, b.name AS target
                LIMIT 100
                """,
                ids=seed_ids,
            )

        triples = [(r["source"], r["relation"], r["target"]) for r in rel_rows]
        if hop_depth == 1:
            neighbor_query = """
                MATCH (seed:Entity)-[*1..1]-(neighbor:Entity)
                WHERE seed.id IN $ids
                RETURN DISTINCT neighbor.id AS nid
            """
        else:
            neighbor_query = """
                MATCH (seed:Entity)-[*1..2]-(neighbor:Entity)
                WHERE seed.id IN $ids
                RETURN DISTINCT neighbor.id AS nid
            """
        neighbor_rows = session.run(neighbor_query, ids=seed_ids)
        all_entity_ids = seed_ids + [r["nid"] for r in neighbor_rows]
 
        doc_rows = session.run(
            """
            MATCH (d:Document)-[:MENTIONS]->(e:Entity)
            WHERE e.id IN $entity_ids
            RETURN DISTINCT d.id AS doc_id
            LIMIT 20
            """,
            entity_ids=all_entity_ids,
        )
        doc_ids = [r["doc_id"] for r in doc_rows]
 
    chunks     = _fetch_chunks_for_docs(doc_ids, top_k=GRAPH_CHUNK_K)
    graph_ctx  = GraphContext(triples=triples, entity_doc_ids=doc_ids)
    return chunks, graph_ctx

def _fetch_chunks_for_docs(doc_ids: list[str], top_k: int) -> list[RetrievedChunk]:
    if not doc_ids:
        return []
 
    conn = psycopg2.connect(POSTGRES_DSN)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT c.id, c.document_id, c.text, d.title, d.source_type
        FROM   chunks c
        JOIN   documents d ON d.id = c.document_id
        WHERE  c.document_id::text = ANY(%s)
        ORDER  BY c.chunk_index ASC
        LIMIT  %s
        """,
        (doc_ids, top_k),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
 
    return [
        RetrievedChunk(
            chunk_id=row[0],
            document_id=row[1],
            text=row[2],
            score=0.0,         
            source="graph",
            title=row[3] or "",
            source_type=row[4] or "",
        )
        for row in rows
    ]

async def retrieve(query: str,driver) -> tuple[list[RetrievedChunk], list[RetrievedChunk], GraphContext, QueryAnalysis]:
    analysis = await analyze_query(query)
 
    loop = asyncio.get_event_loop()
 
    semantic_chunks, (graph_chunks, graph_ctx) = await asyncio.gather(
        semantic_search(analysis),
        loop.run_in_executor(None, graph_search, analysis, driver),
    )
 
    return semantic_chunks, graph_chunks, graph_ctx, analysis