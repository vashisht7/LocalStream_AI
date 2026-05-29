import os
import logging
import hashlib
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from localstream.config import settings

logger = logging.getLogger("localstream.database")

def get_deterministic_id(filename: str, idx: int) -> int:
    """
    Generates a deterministic 64-bit integer ID from a filename and index
    to prevent point ID collisions and support idempotent upserts.
    """
    key = f"{filename}_{idx}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    # 15 hex characters is 60 bits, which fits safely within standard unsigned 64-bit int limits
    return int(h[:15], 16)

class VectorDBManager:
    """Manages the connection, schema setup, and vector indexing operations in Qdrant."""
    
    def __init__(self):
        self.client = None
        # Attempt to connect to a running Qdrant server at localhost:6333 first
        try:
            logger.info(f"Connecting to Qdrant server at {settings.qdrant_url}...")
            self.client = QdrantClient(url=settings.qdrant_url, timeout=3.0)
            # Fetch collections list to verify the connection is active
            self.client.get_collections()
            logger.info("Successfully connected to Qdrant server.")
        except Exception as e:
            logger.warning(
                f"Could not connect to Qdrant server at {settings.qdrant_url}: {e}. "
                f"Falling back to embedded local storage path: {settings.qdrant_data_dir}"
            )
            # Fallback: run embedded Qdrant on local disk storage
            self.client = QdrantClient(path=settings.qdrant_data_dir)
            logger.info("Initialized local disk (embedded) QdrantClient.")

        # Load embedding model locally
        device = settings.compute_device
        logger.info(f"Loading local SentenceTransformer model '{settings.embedding_model_name}' on device '{device}'...")
        try:
            self.model = SentenceTransformer(settings.embedding_model_name, device=device)
            logger.info("SentenceTransformer model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {e}")
            raise RuntimeError(f"Embedding model loading failed: {e}") from e

    def initialize_schema(self):
        """Safely verifies database collections. Creates the collection if missing."""
        collection_name = settings.collection_name
        logger.info(f"Initializing schema for collection: {collection_name}")
        try:
            collections_res = self.client.get_collections()
            existing_collections = [col.name for col in collections_res.collections]
            
            if collection_name not in existing_collections:
                logger.info(f"Collection '{collection_name}' does not exist. Creating collection...")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=384,  # all-MiniLM-L6-v2 outputs 384-dimensional dense vectors
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection '{collection_name}' successfully created.")
            else:
                logger.info(f"Collection '{collection_name}' already exists.")
        except Exception as e:
            logger.error(f"Error during collection initialization: {e}")
            raise RuntimeError(f"Database schema initialization failed: {e}") from e

    def index_media_content(self, raw_file_path: str, transcript_chunks: list):
        """
        Embeds transcript chunks and upserts the payloads into Qdrant collection.
        Uses deterministic IDs to guarantee idempotency and avoid indexing collisions.
        """
        if not transcript_chunks:
            logger.warning(f"No transcript chunks to index for file: {raw_file_path}")
            return

        filename = os.path.basename(raw_file_path)
        logger.info(f"Generating embeddings and indexing {len(transcript_chunks)} chunks for {filename}...")

        # Batch encode all segment texts for performance
        texts = [chunk["text"] for chunk in transcript_chunks]
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e

        points = []
        for idx, chunk in enumerate(transcript_chunks):
            point_id = get_deterministic_id(filename, idx)
            
            # Construct standard payload required for timestamps and source matching
            payload = {
                "filename": filename,
                "file_path": raw_file_path,
                "text": chunk["text"],
                "start_timestamp": chunk["start_timestamp"],
                "end_timestamp": chunk["end_timestamp"],
                "start_seconds": chunk["start_seconds"]
            }
            
            # Embeddings return as NumPy arrays. Convert to regular Python list of floats.
            vector = embeddings[idx].tolist()
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            )

        try:
            self.client.upsert(
                collection_name=settings.collection_name,
                points=points
            )
            logger.info(f"Successfully upserted {len(points)} points to Qdrant collection '{settings.collection_name}'.")
        except Exception as e:
            logger.error(f"Failed to upsert points to Qdrant: {e}")
            raise RuntimeError(f"Database indexing failed: {e}") from e
