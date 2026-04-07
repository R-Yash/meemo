import asyncio
from google import genai
from google.genai import types
from typing import AsyncIterator

from .rag import RetrievedChunk, GraphContext, QueryAnalysis

client = genai.Client()

MAX_CONTEXT_CHARS = 14000  
MAX_KG_TRIPLES    = 25     

def _build_kg_summary(graph_ctx: GraphContext) -> str:
    if not graph_ctx.triples:
        return ""

    lines = ["[Knowledge Graph]\n"]
    for src, rel, tgt in graph_ctx.triples[:MAX_KG_TRIPLES]:
        label = rel.replace("_", " ").title()
        lines.append(f"  • {src}  →[{label}]→  {tgt}")
    return "\n".join(lines)

def assemble_context(chunks: list[RetrievedChunk], graph_ctx: GraphContext) -> str:
    parts: list[str] = []
    budget = MAX_CONTEXT_CHARS

    kg_block = _build_kg_summary(graph_ctx)
    if kg_block:
        parts.append(kg_block)
        budget -= len(kg_block)

    parts.append("\n[Retrieved Passages]\n")

    for i, chunk in enumerate(chunks, 1):
        source_tag = f" [{chunk.source}]" if chunk.source != "semantic" else ""
        header = (
            f"\n--- [{i}] {chunk.title or chunk.document_id} "
            f"({chunk.source_type}){source_tag} | score={chunk.score:.4f} ---\n"
        )
        block = header + chunk.text
        if len(block) > budget:
            parts.append(header + chunk.text[: max(200, budget)] + "\n[…truncated]")
            break
        parts.append(block)
        budget -= len(block)

    return "\n".join(parts)

_SYSTEM_PROMPT = (
    "You are Meemo, a personal knowledge assistant with access to the user's notes, "
    "documents, and a knowledge graph built from their data.\n\n"
    "Rules:\n"
    "- Answer ONLY using the context provided. Do not hallucinate.\n"
    "- If the context is insufficient, say so clearly and explain what's missing.\n"
    "- Cite source titles inline like [Source Title] when making specific claims.\n"
    "- Prefer precise, concise answers. Use bullet points only when listing multiple items.\n"
    "- When the Knowledge Graph block is present, use it to inform relational reasoning "
    "  (e.g. who worked with whom, what caused what) even if the passages are indirect."
)

async def generate_answer(
    query:     str,
    chunks:    list[RetrievedChunk],
    graph_ctx: GraphContext,
    analysis:  QueryAnalysis,
    stream:    bool = True,
) -> str:
    context = assemble_context(chunks, graph_ctx)

    user_prompt = (
        f"{context}\n\n"
        f"{'─' * 60}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    config = types.GenerateContentConfig(
        system_instruction=_SYSTEM_PROMPT,
        temperature=0.2,        
        max_output_tokens=1024,
    )

    if stream:
        return await _stream_answer(user_prompt, config)
    else:
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config=config,
        )
        return response.text


async def _stream_answer(user_prompt: str, config: types.GenerateContentConfig) -> str:
    full_text = ""
    async for chunk in await client.aio.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config=config,
    ):
        if chunk.text:
            print(chunk.text, end="", flush=True)
            full_text += chunk.text
    print()   
    return full_text