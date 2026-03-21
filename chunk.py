import uuid
import logging
import google as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
 
from ingest import Document
 
log = logging.getLogger(__name__)
 
CHUNK_SIZE    = 1500   
CHUNK_OVERLAP = 200    
 
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", "?", "!", " ", ""],
)
 

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


def embed_chunks(chunks: list[dict]) -> list[dict]:
    for chunk in chunks:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=chunk["text"],
            task_type="retrieval_document",
        )
        chunk["embedding"] = result["embedding"]
 
    log.info(f"  Embedded {len(chunks)} chunks")
    return chunks