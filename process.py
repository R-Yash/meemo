import os
import json
import hashlib
import logging
from dotenv import load_dotenv

import psycopg2
from google import genai
from google.genai import types
from pinecone import Pinecone
from neo4j import GraphDatabase

from ingest import Document, extract_note
from chunk import chunk_document, embed_chunks

import asyncio

log = logging.getLogger(__name__)
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_KEY = os.getenv("PINECONE_KEY")
POSTGRES_DSN = os.getenv("POSTGRES_DSN")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

client = genai.Client()

async def generate_tags(doc: Document) -> list[str]:
    prompt = (
        "Generate 3–7 short lowercase tags for the following text.\n\n"
        f"{doc.text[:2000]}"
    )

    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-lite", contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[str], 
        )
    )
    try:
        tags = json.loads(response.text)
        return [t.lower().strip() for t in tags]
    except json.JSONDecodeError:
        log.warning("Tag parsing failed, skipping tags.")
        return []

async def extract_entities(doc: Document) -> list[dict]:
    prompt = (
        "Extract named entities from the text below. "
        "Return an array of entity names."
        "type must be one of: PERSON, PLACE, ORG, CONCEPT, TOPIC. "
        "No explanation, just the array.\n\n"
        f"{doc.text[:3000]}"
    )
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-lite", contents=prompt,
        config=types.GenerateContentConfig(
            response_schema=list[str],
            response_mime_type="application/json", 
        )
    )

    try:
        return list(json.loads(response.text))
    except json.JSONDecodeError:
        log.warning("Entity parsing failed, skipping entities.")
        return []

async def extract_relationships(doc: Document, entities: list[dict]) -> list[dict]:
    if len(entities) < 2:
        return []

    prompt = (
        "Given the following entities extracted from a text, identify meaningful relationships between them. "
        "Return a JSON array of objects with keys 'source', 'target', and 'relation'. "
        "'source' and 'target' should be entity names from the list below. "
        "'relation' should be a short uppercase label like WORKS_AT, FOUNDED, PART_OF, CAUSED_BY, RELATED_TO, etc. "
        "Only return relationships that are clearly supported by the text. "
        "No explanation, just the array.\n\n"
        f"Entities: {entities}\n\n"
        f"Text: {doc.text[:3000]}"
    )
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-lite", contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list, 
        )
    )
    
    try:
        rels = json.loads(response.text)
        return [r for r in rels if "source" in r and "target" in r and "relation" in r]
    except json.JSONDecodeError:
        log.warning("Relationship parsing failed, skipping relationships.")
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
 
    for tag in tags:
        cur.execute(
            """
            INSERT INTO tags (document_id, tag)
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING
            """,
            (doc.doc_id, tag),
        )
 
    conn.commit()
    cur.close()
    conn.close()
    log.info("  Written to Postgres")

def write_to_pinecone(doc: Document, chunks: list[dict]):
    pc = Pinecone(api_key=PINECONE_KEY)
    index = pc.Index(PINECONE_INDEX)
 
    vectors = [
        {
            "id": chunk["chunk_id"],
            "values": chunk["embedding"],
            "metadata": {
                "document_id": doc.doc_id,
                "source_type": doc.source_type,
                "title":       doc.title,
                "text":        chunk["text"],
            },
        }
        for chunk in chunks
    ]
 
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i : i + batch_size])
 
    log.info(f"  Written {len(vectors)} vectors to Pinecone")

def write_to_neo4j(doc: Document, entities: list[dict], relations: list[dict]):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        session.run(
            """
            MERGE (d:Document {id: $id})
            SET d.title = $title, d.source_type = $source_type
            """,
            id=doc.doc_id, title=doc.title, source_type=doc.source_type,
        )

        for ent in entities:
            entity_id = hashlib.sha256(ent.lower().encode()).hexdigest()[:16]
            session.run(
                "MERGE (e:Entity {id: $id}) SET e.name = $name",
                id=entity_id, name=ent,
            )
            session.run(
                """
                MATCH (d:Document {id: $doc_id})
                MATCH (e:Entity {id: $entity_id})
                MERGE (d)-[:MENTIONS]->(e)
                """,
                doc_id=doc.doc_id, entity_id=entity_id,
            )

        for rel in relations:
            source_id = hashlib.sha256(rel["source"].lower().encode()).hexdigest()[:16]
            target_id = hashlib.sha256(rel["target"].lower().encode()).hexdigest()[:16]
            session.run(
                f"""
                MATCH (a:Entity {{id: $source_id}})
                MATCH (b:Entity {{id: $target_id}})
                MERGE (a)-[:`{rel['relation']}`]->(b)
                """,
                source_id=source_id, target_id=target_id,
            )

    driver.close()

async def ingest_async(doc: Document) -> str:
    log.info(f"Ingesting: {doc}")
 
    chunks = chunk_document(doc)

    chunks, tags, entities = await asyncio.gather(
        embed_chunks(chunks),
        generate_tags(doc),
        extract_entities(doc),
    )

    relations = await extract_relationships(doc, entities)

    log.info(f"  Tags: {tags}")
    log.info(f"  Entities: {entities}")
    log.info(f"  Relationships: {relations}")   
 
    write_to_postgres(doc, chunks, tags)
    write_to_pinecone(doc, chunks)
    write_to_neo4j(doc, entities, relations)
 
    log.info(f"  Done → {doc.doc_id}")
    return doc.doc_id


text = """
Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist best known for developing the theory of relativity. 
Einstein also made important contributions to quantum theory. 
His mass–energy equivalence formula E = mc2, which arises from special relativity, has been called "the world's most famous equation".
He received the 1921 Nobel Prize in Physics for "his services to theoretical physics, and especially for his discovery of the law of the 
photoelectric effect", a pivotal step in the development of quantum theory.
Einstein moved to Switzerland in 1895, forsaking his German citizenship the following year. 
In 1897, at the age of seventeen, he enrolled in the mathematics and physics teaching diploma program at the Swiss federal polytechnic school in Zurich, graduating in 1900
Adolf Hitler came to power in Germany.
Horrified by the Nazi persecution of his fellow Jews,he decided to remain in the US, and was granted American citizenship in 1940.
On the eve of World War II, he endorsed a letter to President Franklin D. Roosevelt alerting him to the potential German nuclear weapons program
"""

doc = extract_note(text = text, title="Einstein")
def ingest(doc: Document) -> str:
    return asyncio.run(ingest_async(doc))
ingest(doc)
