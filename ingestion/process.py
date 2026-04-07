import os
import json
import hashlib
from dotenv import load_dotenv

import psycopg2
from google import genai
from google.genai import types
from pinecone import Pinecone
from neo4j import GraphDatabase

from ingestion.ingest import Document, extract_note
from ingestion.chunk import chunk_document, embed_chunks

import asyncio
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
POSTGRES_DSN = os.getenv("POSTGRES_DSN")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

client = genai.Client()

VALID_ENTITY_TYPES = {"PERSON", "PLACE", "ORG", "CONCEPT", "EVENT"}

def entity_id(name: str) -> str:
    return hashlib.sha256(name.strip().lower().encode()).hexdigest()[:16]

async def generate_tags(doc: Document) -> list[str]:
    prompt = (
        "Generate 3–7 short lowercase tags for the following text.\n\n"
        f"{doc.text[:2000]}"
    )
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-lite", contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        )
    )
    try:
        tags = json.loads(response.text)
        # Deduplicate and normalize tags
        unique_tags = list(set(t.lower().strip() for t in tags if t.strip()))
        return unique_tags
    except json.JSONDecodeError:
        return []


async def extract_entities(doc: Document) -> list[dict]:
    prompt = (
        "Extract named entities from the text below. "
        "Return a JSON array of objects each with exactly two keys: "
        "'name' (the entity name as it appears) and "
        "'type' (one of: PERSON, PLACE, ORG, CONCEPT, EVENT). "
        "No explanation, just the array.\n\n"
        f"{doc.text[:3000]}"
    )
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-lite", contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        )
    )
    try:
        raw = json.loads(response.text)
        entities = []
        for item in raw:
            if isinstance(item, dict) and "name" in item:
                entities.append({
                    "name": item["name"].strip(),
                    "type": item.get("type", "CONCEPT").upper(),
                })
            elif isinstance(item, str):
                entities.append({"name": item.strip(), "type": "CONCEPT"})
        return entities
    except json.JSONDecodeError:
        return []


async def canonicalize_entities(entities: list[dict], existing_names: list[str]) -> list[dict]:
    if not existing_names:
        return entities

    raw_names = [e["name"] for e in entities]
    prompt = (
        "You are building a knowledge graph. "
        "Given NEW entity names and EXISTING entity names already in the graph, "
        "map each new name to an existing name if they refer to the same real-world entity "
        "(aliases, abbreviations, alternate spellings). Otherwise keep the new name as-is. "
        "Return ONLY a flat JSON object mapping each new name to its canonical name.\n\n"
        f"New entities: {raw_names}\n"
        f"Existing entities: {existing_names}"
    )
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-lite", contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        )
    )
    try:
        mapping: dict[str, str] = json.loads(response.text)
    except json.JSONDecodeError:
        mapping = {}

    return [
        {"name": mapping.get(e["name"], e["name"]).strip(), "type": e["type"]}
        for e in entities
    ]


async def extract_relationships(doc: Document, entities: list[dict]) -> list[dict]:
    if len(entities) < 2:
        return []

    entity_names = [e["name"] for e in entities]
    prompt = (
        "Given the entities and text below, extract meaningful relationships. "
        "Return a JSON array of objects with keys 'source', 'target', 'relation'. "
        "'source' and 'target' must be names taken verbatim from the entity list. "
        "'relation' must be a short UPPERCASE label like DEVELOPED, RECEIVED, BORN_IN, "
        "LEADER_OF, INVADED, CAUSED_BY, FOUNDED, CONTRIBUTED_TO, NATIONALITY, DISCOVERED, "
        "ALERTED, INITIATED, TARGETED, WORKS_AT, PART_OF, etc. "
        "Include ONLY ONE relationship per (source, target) pair — the most specific one. "
        "Only include relationships explicitly stated in the text. "
        "No explanation, just the array.\n\n"
        f"Entities: {entity_names}\n\n"
        f"Text: {doc.text[:3000]}"
    )
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-lite", contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        )
    )
    try:
        rels = json.loads(response.text)
        valid = [
            r for r in rels
            if isinstance(r, dict)
            and "source" in r and "target" in r and "relation" in r
            and r["source"] in entity_names and r["target"] in entity_names
            and r["source"] != r["target"]
        ]
        seen: set[tuple[str, str]] = set()
        deduped = []
        for r in valid:
            key = (r["source"], r["target"])
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        return deduped
    except json.JSONDecodeError:
        return []


def write_to_postgres(doc: Document, chunks: list[dict], tags: list[str]):
    conn = psycopg2.connect(POSTGRES_DSN)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO documents (id, title, source_type, source_url, raw_text, created_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING
        """,
        (
            doc.doc_id, doc.title, doc.source_type,
            doc.source if doc.source_type in ("youtube", "url") else None,
            doc.text, doc.created_at,
        ),
    )

    for chunk in chunks:
        cur.execute(
            """
            INSERT INTO chunks (id, document_id, chunk_index, text, pinecone_id)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            (chunk["chunk_id"], doc.doc_id, chunk["chunk_index"],
             chunk["text"], chunk["chunk_id"]),
        )

    # Delete existing tags for this document before inserting new ones
    # This ensures tags are replaced on re-ingestion rather than accumulated
    cur.execute("DELETE FROM tags WHERE document_id = %s", (doc.doc_id,))

    for tag in tags:
        cur.execute(
            """
            INSERT INTO tags (document_id, tag)
            VALUES (%s, %s)
            """,
            (doc.doc_id, tag),
        )

    conn.commit()
    cur.close()
    conn.close()


def write_to_pinecone(doc: Document, chunks: list[dict]):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)

    vectors = [
        {
            "id": chunk["chunk_id"],
            "values": chunk["embedding"],
            "metadata": {
                "document_id": doc.doc_id,
                "source_type": doc.source_type,
                "title": doc.title,
                "text": chunk["text"],
            },
        }
        for chunk in chunks
    ]

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i: i + batch_size])


def write_to_neo4j(doc: Document, entities: list[dict], relations: list[dict], driver):
    with driver.session() as session:
        session.run(
            """
            MERGE (d:Document {id: $id})
            SET d.title = $title, d.source_type = $source_type
            """,
            id=doc.doc_id, title=doc.title, source_type=doc.source_type,
        )

        for ent in entities:
            name = ent["name"]
            ent_type = ent["type"] if ent["type"] in VALID_ENTITY_TYPES else "CONCEPT"
            eid = entity_id(name)

            session.run(
                """
                MERGE (e:Entity {id: $id})
                SET e.name = $name, e.type = $type
                WITH e
                CALL apoc.create.addLabels(e, [$type]) YIELD node
                RETURN node
                """,
                id=eid, name=name, type=ent_type,
            )

            session.run(
                """
                MATCH (d:Document {id: $doc_id})
                MATCH (e:Entity {id: $entity_id})
                MERGE (d)-[:MENTIONS]->(e)
                """,
                doc_id=doc.doc_id, entity_id=eid,
            )

        for rel in relations:
            source_eid = entity_id(rel["source"])
            target_eid = entity_id(rel["target"])
            relation = rel["relation"].upper().replace(" ", "_")

            session.run(
                f"""
                MATCH (a:Entity {{id: $source_id}})
                MATCH (b:Entity {{id: $target_id}})
                MERGE (a)-[:`{relation}`]->(b)
                """,
                source_id=source_eid, target_id=target_eid,
            )


async def ingest_async(doc: Document, driver) -> str:
    chunks = chunk_document(doc)

    chunks, tags, raw_entities = await asyncio.gather(
        embed_chunks(chunks),
        generate_tags(doc),
        extract_entities(doc),
    )

    with driver.session() as session:
        result = session.run("MATCH (e:Entity) RETURN e.name AS name")
        existing_names = [r["name"] for r in result]
    
    entities = await canonicalize_entities(raw_entities, existing_names)

    relations = await extract_relationships(doc, entities)

    write_to_postgres(doc, chunks, tags)
    write_to_pinecone(doc, chunks)
    write_to_neo4j(doc, entities, relations, driver)