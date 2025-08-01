import os
import yaml
import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
COLLECTION_NAME = "codebase"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_ENDPOINT = os.getenv("LMSTUDIO_API", "http://localhost:1234/v1/embeddings")

client = QdrantClient(url=QDRANT_HOST)

def load_config():
    with open("indexer/config.yaml", "r") as f:
        return yaml.safe_load(f)

def get_embedding(text: str) -> list[float]:
    payload = {
        "input": [text],
        "model": EMBEDDING_MODEL
    }
    response = httpx.post(EMBEDDING_ENDPOINT, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

def should_ignore(path: Path, ignore_patterns: list[str]) -> bool:
    return any(str(path).startswith(pattern) for pattern in ignore_patterns)

def index_codebase():
    config = load_config()
    base_path = Path(config["project_path"])
    ignore = config.get("ignore", [])

    files = list(base_path.rglob("*.ts")) + list(base_path.rglob("*.tsx"))

    # Создаём коллекцию, если не существует
    if COLLECTION_NAME not in [col.name for col in client.get_collections().collections]:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
        )

    for file_path in tqdm(files, desc="Indexing"):
        if should_ignore(file_path, ignore):
            continue
        try:
            content = file_path.read_text(encoding="utf-8")
            embedding = get_embedding(content)
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=[models.PointStruct(
                    id=hash(file_path),
                    vector=embedding,
                    payload={"path": str(file_path), "text": content}
                )]
            )
        except Exception as e:
            print(f"❌ Failed on {file_path}: {e}")

if __name__ == "__main__":
    index_codebase()
