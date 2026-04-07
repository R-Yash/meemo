import json
import asyncio
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from neo4j import Driver

from app.db import get_neo4j
from app.models import QueryRequest, QueryResponse, QueryAnalysisOut, RetrievedChunkOut

from  retrieval.rag import retrieve
from retrieval.reranker import reciprocal_rank_fusion
from retrieval.answer import assemble_context, _SYSTEM_PROMPT
from retrieval.answer import generate_answer

from google import genai
from google.genai import types

router = APIRouter(prefix="/query", tags=["query"])
client = genai.Client()

def _sse(event: str, data) -> str:
    payload = data if isinstance(data, str) else json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"

def _sse_error(message: str) -> str:
    return _sse("error", {"detail": message})

async def _stream_query(query: str, driver: Driver, verbose: bool):
    try:
        semantic_chunks, graph_chunks, graph_ctx, analysis = await retrieve(query, driver)
        fused_chunks = reciprocal_rank_fusion(semantic_chunks, graph_chunks)

        yield _sse("analysis", {
            "original":  analysis.original,
            "rewritten": analysis.rewritten,
            "entities":  analysis.entities,
            "tags":      analysis.tags,
        })

        for i, c in enumerate(fused_chunks):
            yield _sse(f"chunk_{i}", {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "title": c.title,
                "source_type": c.source_type,
                "source": c.source,
                "score": c.score,
                "text": c.text,
            })

        context     = assemble_context(fused_chunks, graph_ctx)
        user_prompt = (
            f"{context}\n\n"
            f"{'─' * 60}\n\n"
            f"Question: {query}\n\nAnswer:"
        )
        config = types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=1024,
        )

        full_answer = ""
        async for chunk in await client.aio.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config=config,
        ):
            if chunk.text:
                full_answer += chunk.text
                yield _sse("token", chunk.text)

        yield _sse("done", {
            "answer":  full_answer,
            "triples": graph_ctx.triples,
        })

    except Exception as exc:
        yield _sse_error(str(exc))


async def _batch_query(query: str, driver: Driver, verbose: bool) -> QueryResponse:
    semantic_chunks, graph_chunks, graph_ctx, analysis = await retrieve(
        query, driver
    )
    fused_chunks = reciprocal_rank_fusion(semantic_chunks, graph_chunks)

    answer = await generate_answer(
        query=query,
        chunks=fused_chunks,
        graph_ctx=graph_ctx,
        analysis=analysis,
        stream=False,
    )

    return QueryResponse(
        answer=answer,
        analysis=QueryAnalysisOut(
            original=analysis.original,
            rewritten=analysis.rewritten,
            entities=analysis.entities,
            tags=analysis.tags,
        ),
        chunks=[
            RetrievedChunkOut(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                text=c.text,
                score=c.score,
                source=c.source,
                title=c.title,
                source_type=c.source_type,
            )
            for c in fused_chunks
        ],
        triples=graph_ctx.triples,
    )

@router.post("/")
async def query_endpoint(body: QueryRequest, driver: Driver = Depends(get_neo4j)):

    if body.stream:
        return StreamingResponse(
            _stream_query(body.query, driver, body.verbose),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",   
            },
        )

    try:
        return await _batch_query(body.query, driver, body.verbose)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc