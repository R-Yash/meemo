from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, HttpUrl, field_validator

class IngestNoteRequest(BaseModel):
    text: str
    title: str = ""

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        return v

class IngestUrlRequest(BaseModel):
    url: str
    title: str = ""

class IngestYouTubeRequest(BaseModel):
    url:   str
    title: str = ""

    @field_validator("url")
    @classmethod
    def must_be_youtube(cls, v: str) -> str:
        if "youtube.com/watch" not in v and "youtu.be/" not in v:
            raise ValueError("URL does not look like a YouTube video link")
        return v

class IngestResponse(BaseModel):
    doc_id: str
    title: str
    source_type: str
    chunks: int
    tags: list[str]
    message: str = "Ingested successfully"

class QueryRequest(BaseModel):
    query: str
    stream: bool = True
    verbose: bool = False

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty")
        return v

class QueryAnalysisOut(BaseModel):
    original: str
    rewritten: str
    entities: list[str]
    tags: list[str]

class RetrievedChunkOut(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    source: Literal["semantic", "graph", "both"]
    title: str
    source_type: str

class QueryResponse(BaseModel):
    answer: str
    analysis: QueryAnalysisOut
    chunks: list[RetrievedChunkOut]
    triples: list[tuple[str, str, str]]

class DocumentOut(BaseModel):
    doc_id: str
    title: str
    source_type: str
    source_url: Optional[str]
    created_at: str
    tags: list[str] = []
    chunk_count: int = 0

class DocumentDetail(DocumentOut):
    raw_text: str

class ChunkOut(BaseModel):
    chunk_id: str
    chunk_index: int
    text: str

class DocumentListResponse(BaseModel):
    documents: list[DocumentOut]
    total: int
    page: int
    page_size: int

class DeleteResponse(BaseModel):
    doc_id: str
    deleted: bool
    message: str

class TagListResponse(BaseModel):
    tags: list[str]

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    postgres: bool
    pinecone: bool
    neo4j: bool