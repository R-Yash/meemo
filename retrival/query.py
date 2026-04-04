import os
import asyncio
from dotenv import load_dotenv
from neo4j import GraphDatabase

from rag import retrieve
from reranker import reciprocal_rank_fusion, explain_ranking
from answer import generate_answer

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


async def query_meemo(user_query: str, stream:  bool = True, verbose: bool = False) -> str:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        semantic_chunks, graph_chunks, graph_ctx, analysis = await retrieve(user_query, driver)

        if verbose:
            print(f"\n[Query Analysis]")
            print(f"  Rewritten : {analysis.rewritten}")
            print(f"  Entities  : {analysis.entities}")
            print(f"  Tags      : {analysis.tags}")
            print(f"\n[Retrieval]")
            print(f"  Semantic  : {len(semantic_chunks)} chunks")
            print(f"  Graph     : {len(graph_chunks)} chunks from {len(graph_ctx.entity_doc_ids)} docs")
            print(f"  KG triples: {len(graph_ctx.triples)}")

        fused_chunks = reciprocal_rank_fusion(semantic_chunks, graph_chunks)

        if verbose:
            print(f"\n{explain_ranking(fused_chunks)}\n")

        if stream:
            print(f"\n[Meemo]\n")

        answer = await generate_answer(
            query=user_query,
            chunks=fused_chunks,
            graph_ctx=graph_ctx,
            analysis=analysis,
            stream=stream,
        )

        return answer

    finally:
        driver.close()


if __name__ == "__main__":
    import sys

    query   = "What did Einstein discover"

    asyncio.run(query_meemo(query, stream=True))