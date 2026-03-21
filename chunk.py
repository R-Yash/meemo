import os
import uuid
import logging
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import asyncio

from ingest import Document

load_dotenv()
log = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHUNK_SIZE    = 1500   
CHUNK_OVERLAP = 200    
 
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", "?", "!", " ", ""],
)

client = genai.Client()

def chunk_document(doc: Document) -> list[dict]:
    texts = _splitter.split_text(doc.text)

    chunks = [
        {
            "chunk_id":    str(uuid.uuid4()),
            "document_id": doc.doc_id,
            "chunk_index": i,
            "text":        text,
        }
        for i, text in enumerate(texts)
    ]
 
    log.info(f"  {len(chunks)} chunks created for doc {doc.doc_id!r}")
    return chunks

async def _embed_one(chunk: dict) -> dict:
    result = await client.aio.models.embed_content(
        model="gemini-embedding-2-preview",
        contents=chunk["text"],
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY",output_dimensionality=768),
    )
    chunk["embedding"] = result.embeddings[0].values
    return chunk

async def embed_chunks(chunks: list[dict]) -> list[dict]:
    return await asyncio.gather(*[_embed_one(chunk) for chunk in chunks])

# def embed_chunks(chunks: list[dict]) -> list[dict]:
#     return asyncio.run(embed_chunks_async(chunks))

