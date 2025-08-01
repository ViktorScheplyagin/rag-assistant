import os
from pathlib import Path
from typing import List, Tuple

import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from indexer.graph_utils import get_related_files


QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
COLLECTION_NAME = "codebase"
LMSTUDIO_API = os.getenv("LMSTUDIO_API", "http://localhost:1234")
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "gemma:3n"


app = FastAPI(title="RAG Assistant")


class RagRequest(BaseModel):
    question: str
    target_file: str
    depth: int = 1


class RagResponse(BaseModel):
    answer: str


def _load_file_payload(client: QdrantClient, path: str) -> Tuple[str, str] | None:
    """Return (path, text) from Qdrant payload if available."""
    res, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=qmodels.Filter(
            must=[qmodels.FieldCondition(key="path", match=qmodels.MatchValue(value=path))]
        ),
        limit=1,
    )
    if res:
        payload = res[0].payload or {}
        text = payload.get("text")
        if text is not None:
            return path, text
    return None


@app.post("/rag", response_model=RagResponse)
async def rag_endpoint(req: RagRequest) -> RagResponse:
    client = QdrantClient(url=QDRANT_HOST)
    async with httpx.AsyncClient() as http_client:
        # 1. embed question
        embed_payload = {"input": [req.question], "model": EMBEDDING_MODEL}
        embed_resp = await http_client.post(
            f"{LMSTUDIO_API}/v1/embeddings", json=embed_payload, timeout=30
        )
        embed_resp.raise_for_status()
        question_emb = embed_resp.json()["data"][0]["embedding"]

    # 2. search in qdrant
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=question_emb,
        limit=10,
    )

    seen = set()
    context: List[Tuple[str, str]] = []
    for hit in results:
        payload = hit.payload or {}
        path = payload.get("path")
        text = payload.get("text")
        if path and text and path not in seen:
            seen.add(path)
            context.append((path, text))

    # 3. get related files
    related_files = get_related_files(req.target_file, depth=req.depth)
    for rel_path in related_files:
        norm_path = Path(rel_path).as_posix()
        if norm_path in seen:
            continue
        payload = _load_file_payload(client, norm_path)
        if payload:
            seen.add(norm_path)
            context.append(payload)

    # also ensure target file itself is included
    target_norm = Path(req.target_file).as_posix()
    if target_norm not in seen:
        payload = _load_file_payload(client, target_norm)
        if payload:
            seen.add(target_norm)
            context.append(payload)

    # 4. build prompt
    context_parts = [f"--- файл: {path}\n{text}" for path, text in context]
    prompt = (
        f"Вопрос: {req.question}\n\nКонтекст:\n" +
        "\n".join(context_parts) +
        "\n\nОтвет:"
    )
    # 5. send to LM Studio
    chat_payload = {
        "model": CHAT_MODEL,
        "temperature": 0.4,
        "messages": [{"role": "user", "content": prompt}],
    }
    async with httpx.AsyncClient() as http_client:
        chat_resp = await http_client.post(
            f"{LMSTUDIO_API}/v1/chat/completions", json=chat_payload, timeout=60
        )
        chat_resp.raise_for_status()
        answer = chat_resp.json()["choices"][0]["message"]["content"]

    return RagResponse(answer=answer)


@app.post("/v1/chat/completions")
async def zed_proxy(request: dict) -> dict:
    """
    OpenAI-compatible endpoint for Zed IDE.
    Expects 'messages', 'model' and optionally 'editor_context'.
    """
    messages = request.get("messages", [])
    question = messages[-1]["content"] if messages else ""
    editor_context = request.get("editor_context", {})
    target_file = editor_context.get("current_file", "index.ts")  # fallback
    
    rag_req = RagRequest(question=question, target_file=target_file)
    response = await rag_endpoint(rag_req)
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": response.answer
                }
            }
        ]
    }
