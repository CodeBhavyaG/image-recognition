import chromadb
from schemas import ImageUnderstanding
from typing import List, Dict, Any

class VectorDB:
    def __init__(self, collection_name: str = "image_embeddings", persist_directory: str = "./data/chroma"):
        # We use a persistent client so embeddings are saved across runs
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # We use default embedding function (all-MiniLM-L6-v2) under the hood
        # Using cosine similarity for distance
        self.collection = self.client.get_or_create_collection(
            name=collection_name, 
            metadata={"hnsw:space": "cosine"}
        )

    def add_image_understanding(self, image_id: str, understanding: ImageUnderstanding):
        """
        Takes the validated ImageUnderstanding schema (from groq_extract.py) 
        and stores it in the vector database.
        """
        # Embed caption + semantic tags to help the matching algorithm
        attrs = ", ".join(understanding.attributes)
        doc = f"Subject: {understanding.subject}. Category: {understanding.category}. Caption: {understanding.caption}. Attributes: {attrs}"
        
        # Store metadata for the mismatch guard logic
        meta = {
            "subject": understanding.subject,
            "category": understanding.category,
            "caption": understanding.caption,
            "confidence": float(understanding.confidence)
        }

        # Upsert allows adding or updating
        self.collection.upsert(
            ids=[image_id],
            documents=[doc],
            metadatas=[meta]
        )

    def rank_images_for_post(self, post_text: str, top_k: int = 5, max_cosine_distance: float = 0.5) -> List[Dict[str, Any]]:
        """
        Ranks images against a blog post and applies a mismatch guard.
        Returns a list of image candidates with their acceptance status.
        """
        results = self.collection.query(
            query_texts=[post_text],
            n_results=top_k
        )

        ranked = []
        if not results["ids"] or not results["ids"][0]:
            return ranked

        # results is a dict with lists of lists for distances, metadatas, ids, documents
        for i in range(len(results["ids"][0])):
            img_id = results["ids"][0][i]
            distance = results["distances"][0][i]
            meta = results["metadatas"][0][i]
            
            # Start evaluating mismatch guard conditions

            # Mismatch guard 1: similarity threshold
            if distance > max_cosine_distance:
                ranked.append({
                    "id": img_id,
                    "status": "REJECTED",
                    "reason": f"Similarity below threshold (cosine distance {distance:.3f} > {max_cosine_distance})",
                    "metadata": meta,
                    "distance": distance
                })
                continue
                
            # Mismatch guard 2: minimum confidence from vision model
            if meta.get("confidence", 0.0) < 0.7:
                 ranked.append({
                    "id": img_id,
                    "status": "REJECTED",
                    "reason": f"Vision model confidence too low ({meta.get('confidence')} < 0.7)",
                    "metadata": meta,
                    "distance": distance
                })
                 continue

            ranked.append({
                "id": img_id,
                "status": "ACCEPTED",
                "reason": "Good match",
                "metadata": meta,
                "distance": distance
            })

        return ranked
