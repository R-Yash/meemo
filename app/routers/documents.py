from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from neo4j import Driver
from pinecone import Pinecone

from app.db import get_neo4j, get_pinecone, PgConn
from app.models import (
    DocumentOut,
    DocumentDetail,
    DocumentListResponse,
    ChunkOut,
    DeleteResponse,
    TagListResponse,
)

router = APIRouter(prefix="/documents", tags=["documents"])

@router.get("/", response_model=DocumentListResponse)
def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    tag: Optional[str]  = Query(None, description="Filter by tag"),
    source_type: Optional[str]  = Query(None, description="Filter by source type"),
    search: Optional[str]  = Query(None, description="Title keyword search"),
):

    offset = (page - 1) * page_size

    filters = ["1=1"]
    params: list = []

    if tag:
        filters.append("d.id IN (SELECT document_id FROM tags WHERE tag = %s)")
        params.append(tag)
    if source_type:
        filters.append("d.source_type = %s")
        params.append(source_type)
    if search:
        filters.append("d.title ILIKE %s")
        params.append(f"%{search}%")

    where = " AND ".join(filters)

    with PgConn() as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM documents d WHERE {where}", params)
        total = cur.fetchone()[0]

        cur.execute(
            f"""
            SELECT d.id, d.title, d.source_type, d.source_url, d.created_at,
                   COALESCE(array_agg(t.tag) FILTER (WHERE t.tag IS NOT NULL), ARRAY[]::text[]) AS tags,
                   COUNT(c.id) AS chunk_count
            FROM   documents d
            LEFT   JOIN tags   t ON t.document_id = d.id
            LEFT   JOIN chunks c ON c.document_id = d.id
            WHERE  {where}
            GROUP  BY d.id, d.title, d.source_type, d.source_url, d.created_at
            ORDER  BY d.created_at DESC
            LIMIT  %s OFFSET %s
            """,
            [*params, page_size, offset],
        )
        rows = cur.fetchall()
        cur.close()

    documents = [
        DocumentOut(
            doc_id=row[0],
            title=row[1] or "",
            source_type=row[2] or "",
            source_url=row[3],
            created_at=str(row[4]),
            tags=list(row[5]) if row[5] else [],
            chunk_count=row[6],
        )
        for row in rows
    ]

    return DocumentListResponse(
        documents=documents,
        total=total,
        page=page,
        page_size=page_size,
    )

@router.get("/{doc_id}", response_model=DocumentDetail)
def get_document(doc_id: str):
    with PgConn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT d.id, d.title, d.source_type, d.source_url, d.created_at, d.raw_text,
                   COALESCE(array_agg(t.tag) FILTER (WHERE t.tag IS NOT NULL), ARRAY[]::text[]),
                   COUNT(c.id)
            FROM   documents d
            LEFT   JOIN tags   t ON t.document_id = d.id
            LEFT   JOIN chunks c ON c.document_id = d.id
            WHERE  d.id = %s
            GROUP  BY d.id
            """,
            (doc_id,),
        )
        row = cur.fetchone()
        cur.close()

    if not row:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentDetail(
        doc_id=row[0],
        title=row[1] or "",
        source_type=row[2] or "",
        source_url=row[3],
        created_at=str(row[4]),
        raw_text=row[5] or "",
        tags=list(row[6]) if row[6] else [],
        chunk_count=row[7],
    )

@router.get("/{doc_id}/chunks", response_model=list[ChunkOut])
def get_document_chunks(doc_id: str):
    with PgConn() as conn:
        cur = conn.cursor()

        cur.execute("SELECT 1 FROM documents WHERE id = %s", (doc_id,))
        if not cur.fetchone():
            cur.close()
            raise HTTPException(status_code=404, detail="Document not found")

        cur.execute(
            """
            SELECT id, chunk_index, text
            FROM   chunks
            WHERE  document_id = %s
            ORDER  BY chunk_index ASC
            """,
            (doc_id,),
        )
        rows = cur.fetchall()
        cur.close()

    return [
        ChunkOut(chunk_id=r[0], chunk_index=r[1], text=r[2])
        for r in rows
    ]

@router.get("/{doc_id}/graph")
def get_document_graph(
    doc_id: str,
    driver: Driver = Depends(get_neo4j),
):
    with PgConn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM documents WHERE id = %s", (doc_id,))
        if not cur.fetchone():
            cur.close()
            raise HTTPException(status_code=404, detail="Document not found")
        cur.close()

    with driver.session() as session:
        entity_rows = session.run(
            """
            MATCH (d:Document {id: $doc_id})-[:MENTIONS]->(e:Entity)
            RETURN e.id AS id, e.name AS name, e.type AS type
            """,
            doc_id=doc_id,
        )
        entities = [
            {"id": r["id"], "name": r["name"], "type": r["type"]}
            for r in entity_rows
        ]

        entity_ids = [e["id"] for e in entities]
        if not entity_ids:
            return {"entities": [], "relations": []}

        rel_rows = session.run(
            """
            MATCH (a:Entity)-[r]->(b:Entity)
            WHERE a.id IN $ids AND b.id IN $ids
            RETURN a.id AS source_id, type(r) AS relation, b.id AS target_id
            """,
            ids=entity_ids,
        )
        relations = [
            {"source": r["source_id"], "relation": r["relation"], "target": r["target_id"]}
            for r in rel_rows
        ]

    return {"entities": entities, "relations": relations}

@router.delete("/{doc_id}", response_model=DeleteResponse)
def delete_document(
    doc_id:  str,
    driver:  Driver   = Depends(get_neo4j),
    pc:      Pinecone = Depends(get_pinecone),
):
    with PgConn() as conn:
        cur = conn.cursor()

        cur.execute("SELECT id FROM documents WHERE id = %s", (doc_id,))
        if not cur.fetchone():
            cur.close()
            raise HTTPException(status_code=404, detail="Document not found")

        cur.execute("SELECT pinecone_id FROM chunks WHERE document_id = %s", (doc_id,))
        pinecone_ids = [r[0] for r in cur.fetchall()]

        cur.execute("DELETE FROM tags   WHERE document_id = %s", (doc_id,))
        cur.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
        cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        cur.close()

    if pinecone_ids:
        try:
            import os
            index = pc.Index(os.environ["PINECONE_INDEX"])
            # Pinecone delete accepts up to 1000 IDs per call
            batch_size = 1000
            for i in range(0, len(pinecone_ids), batch_size):
                index.delete(ids=pinecone_ids[i: i + batch_size])
        except Exception as exc:
            pass

    try:
        with driver.session() as session:
            session.run(
                """
                MATCH (d:Document {id: $doc_id})
                DETACH DELETE d
                """,
                doc_id=doc_id,
            )
    except Exception as exc:
        pass

    return DeleteResponse(
        doc_id=doc_id,
        deleted=True,
        message=f"Document {doc_id!r} removed from all stores",
    )

@router.get("/tags/all", response_model=TagListResponse, tags=["tags"])
def list_tags(min_count: int = Query(1, ge=1)):
    with PgConn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT tag
            FROM   tags
            GROUP  BY tag
            HAVING COUNT(DISTINCT document_id) >= %s
            ORDER  BY COUNT(DISTINCT document_id) DESC
            """,
            (min_count,),
        )
        tags = [r[0] for r in cur.fetchall()]
        cur.close()

    return TagListResponse(tags=tags)