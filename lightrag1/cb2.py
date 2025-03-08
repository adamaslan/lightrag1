# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/cb2.ipynb.

# %% auto 0
__all__ = ['WORKING_DIR', 'load_or_compute_embeddings', 'initialize_rag', 'print_stream', 'main', 'print_query_metrics']

# %% ../nbs/cb2.ipynb 2
import os
import json

def load_or_compute_embeddings(compute_embeddings):
    """
    Checks for required precomputed files and either loads them or computes and saves them.
    
    Parameters:
      compute_embeddings (callable): A function that computes and returns the embeddings.
    
    Returns:
      dict: The loaded or newly computed embeddings.
    """
    required_files = [
        "graph_chunk_entity_relation.graphml",
        "kv_store_text_chunks.json",
        "kv_store_doc_status.json",
        "vdb_chunks.json",
        "kv_store_full_docs.json",
        "vdb_entities.json",
        "kv_store_llm_response_cache.json",
        "vdb_relationships.json"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("Missing files detected:", missing_files)
        # Compute embeddings using the provided function
        embeddings = compute_embeddings()
        # Save results to one or more of the required files
        # (You may need to adapt this to your file structure)
        with open("vdb_chunks.json", "w", encoding="utf-8") as f:
            json.dump(embeddings, f)
        # Save other files as needed...
    else:
        print("All precomputed files found. Loading cached embeddings...")
        with open("vdb_chunks.json", "r", encoding="utf-8") as f:
            embeddings = json.load(f)
    
    return embeddings


# %% ../nbs/cb2.ipynb 3
import asyncio
import nest_asyncio
import os
import inspect
import logging
import csv

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

nest_asyncio.apply()

WORKING_DIR = "./dickens"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="deepseek-r1:1.5b1a",
        llm_model_max_async=4,
        llm_model_max_token_size=32768,
        llm_model_kwargs={
            "host": "http://localhost:11434",
            "options": {"num_ctx": 32768},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts, embed_model="nomic-embed-text", host="http://localhost:11434"
            ),
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def print_stream(stream):
    async for chunk in stream:
        print(chunk, end="", flush=True)


def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())

    # Read CSV data and build a formatted string
    csv_data = ""
    with open("./42a.csv", "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            csv_data += f"Topic: {row['Topic']}\n"
            csv_data += f"Key Concepts/Themes: {row['Key Concepts/Themes']}\n\n"

    # Insert the CSV data into the RAG system
    rag.insert(csv_data)

    # Test different query modes with updated questions

    print("\nNaive Search:")
    print(
        rag.query(
            "What are the main topics and their associated themes?",
            param=QueryParam(mode="naive")
        )
    )

    print("\nLocal Search:")
    print(
        rag.query(
            "Which topic emphasizes the discipline of sensory focus and aesthetic enhancement?",
            param=QueryParam(mode="local")
        )
    )

    print("\nGlobal Search:")
    print(
        rag.query(
            "How do the themes reflect ideas of personal growth and moderation across different topics?",
            param=QueryParam(mode="global")
        )
    )

    print("\nHybrid Search:")
    print(
        rag.query(
            "Can you summarize how each topic combines concepts of self-expression, beauty, and resource management?",
            param=QueryParam(mode="hybrid")
        )
    )

    # Stream response for one of the queries
    resp = rag.query(
        "Can you summarize how each topic combines concepts of self-expression, beauty, and resource management?",
        param=QueryParam(mode="hybrid", stream=True),
    )

    if inspect.isasyncgen(resp):
        asyncio.run(print_stream(resp))
    else:
        print(resp)


if __name__ == "__main__":
    main()


# %% ../nbs/cb2.ipynb 5
# Assuming the `rag` instance has already been created in a previous cell

print("\nAdditional Questions:")

# Naive mode query: exploring economic aspects
print("Naive Query:")
print(
    rag.query(
        "What topic discusses economic prosperity and its impact on community life?",
        param=QueryParam(mode="naive")
    )
)

# Local mode query: connecting artistic expression with systematic resource allocation
print("\nLocal Query:")
print(
    rag.query(
        "Which topic integrates artistic expression with systematic resource allocation?",
        param=QueryParam(mode="local")
    )
)

# Global mode query: comprehensive analysis of aesthetic themes
print("\nGlobal Query:")
print(
    rag.query(
        "Can you provide a comprehensive analysis of aesthetic themes across all topics?",
        param=QueryParam(mode="global")
    )
)


# %% ../nbs/cb2.ipynb 6
import time

def print_query_metrics(query_text, mode):
    start_time = time.time()
    result = rag.query(query_text, param=QueryParam(mode=mode))
    duration = time.time() - start_time
    # Check if the result is an async generator; if so, run the stream synchronously for metrics.
    if hasattr(result, '__aiter__'):
        async def get_full_response():
            response = ""
            async for chunk in result:
                response += chunk
            return response
        result = asyncio.run(get_full_response())
    print(f"Query: {query_text}")
    print(f"Mode: {mode}")
    print("Response:", result)
    print(f"Time taken: {duration:.2f} seconds\n")

print("\nMetrics for Additional Queries:")

# Example metric measurement for a naive mode query
print_query_metrics(
    "How does the system interpret the integration of self-expression with economic factors?", 
    "naive"
)

# Example metric measurement for a global mode query
print_query_metrics(
    "Can you detail the thematic evolution in topics emphasizing moderation?", 
    "global"
)

