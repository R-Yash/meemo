import asyncio
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from neo4j import Driver

from app.db import get_neo4j, PgConn
from app.models import (
    IngestNoteRequest,
    IngestUrlRequest,
    IngestYouTubeRequest,
    IngestResponse,
)

from ingestion.ingest import extract_note, extract_url, extract_youtube, extract_pdf, extract_docx
from ingestion.process import ingest_async

router = APIRouter(prefix="/ingest", tags=["ingest"])

ALLOWED_MIME_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}
MAX_FILE_BYTES = 50 * 1024 * 1024   # 50 MB

async def _run_ingest(doc, driver: Driver) -> IngestResponse:
    try:
        result = await ingest_async(doc, driver)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    with PgConn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT tag FROM tags WHERE document_id = %s", (doc.doc_id,))
        tags = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT COUNT(*) FROM chunks WHERE document_id = %s", (doc.doc_id,))
        chunk_count = cur.fetchone()[0]
        cur.close()

    return IngestResponse(
        doc_id=doc.doc_id,
        title=doc.title,
        source_type=doc.source_type,
        chunks=chunk_count,
        tags=tags,
    )

@router.post("/note", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_note(body: IngestNoteRequest, driver: Driver = Depends(get_neo4j)):
    doc = extract_note(text=body.text, title=body.title)
    return await _run_ingest(doc, driver)

@router.post("/url", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_url(body: IngestUrlRequest, driver: Driver = Depends(get_neo4j)):
    try:
        doc = await asyncio.get_event_loop().run_in_executor(None, extract_url, str(body.url))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not fetch URL: {exc}")

    if body.title:
        doc.title = body.title

    return await _run_ingest(doc, driver)

@router.post("/youtube", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_youtube(body: IngestYouTubeRequest, driver: Driver = Depends(get_neo4j)):
    try:
        doc = await asyncio.get_event_loop().run_in_executor(None, extract_youtube, body.url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"YouTube extraction failed: {exc}")

    if body.title:
        doc.title = body.title

    return await _run_ingest(doc, driver)

@router.post("/file", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_file(file: UploadFile = File(...), title: str = Form(""), driver: Driver = Depends(get_neo4j)):
    file_type = ALLOWED_MIME_TYPES.get(file.content_type)
    if not file_type:
        suffix = Path(file.filename or "").suffix.lower().lstrip(".")
        if suffix not in ("pdf", "docx"):
            raise HTTPException(status_code=415, detail="Only PDF and DOCX files are supported. More Supported formats coming soon!")
        file_type = suffix

    content = await file.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413,detail=f"File exceeds 50 MB limit ({len(content) // 1_048_576} MB received)")

    suffix_map = {"pdf": ".pdf", "docx": ".docx"}
    with tempfile.NamedTemporaryFile(suffix=suffix_map[file_type], delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        loop = asyncio.get_event_loop()
        extractor = extract_pdf if file_type == "pdf" else extract_docx
        doc = await loop.run_in_executor(None, extractor, tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"File parsing failed: {exc}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if title:
        doc.title = title
    elif not doc.title:
        doc.title = Path(file.filename or "untitled").stem

    return await _run_ingest(doc, driver)