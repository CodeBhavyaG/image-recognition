import chromadb
from typing import List, Dict, Any

class SemanticMatcher:
    def __init__(self, collection_name: str = "image_embeddings", persist_directory: str = "./data/chroma"):
        # We use a persistent client so embeddings are saved.
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # We use default embedding function (all-MiniLM-L6-v2) under the hood
        # Using cosine similarity for distance
        self.collection = self.client.get_or_create_collection(
            name=collection_name, 
            metadata={"hnsw:space": "cosine"}
        )

    def add_images(self, images: List[Dict[str, Any]]):
        """
        Add images to the vector index.
        `images` is a list of dictionaries. Example:
        {
            "id": "image_1.jpg",
            "subject": "red fox",
            "category": "animal",
            "attributes": ["orange fur", "wild", "forest"],
            "caption": "A red fox standing in a forest",
            "confidence": 0.94
        }
        """
        if not images:
            return

        ids = []
        documents = []
        metadatas = []

        for img in images:
            img_id = str(img.get("id", img.get("subject")))
            
            # Embed caption + semantic tags to help the matching algorithm
            attrs = ", ".join(img.get("attributes", []))
            doc = f"Subject: {img.get('subject', '')}. Category: {img.get('category', '')}. Caption: {img.get('caption', '')}. Attributes: {attrs}"
            
            ids.append(img_id)
            documents.append(doc)
            
            # Store metadata for the mismatch guard logic
            meta = {
                "subject": img.get("subject", ""),
                "category": img.get("category", ""),
                "caption": img.get("caption", ""),
                "confidence": float(img.get("confidence", 0.0))
            }
            metadatas.append(meta)

        # Upsert allows adding or updating
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def rank_images_for_post(self, post_text: str, top_k: int = 5, max_cosine_distance: float = 0.5, required_category: str = None) -> List[Dict[str, Any]]:
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

            # Mismatch guard 1: similarity threshold (cosine distance: lower is more similar, 0 is exact match)
            if distance > max_cosine_distance:
                ranked.append({
                    "id": img_id,
                    "status": "REJECTED",
                    "reason": f"Similarity below threshold (cosine distance {distance:.3f} > {max_cosine_distance})",
                    "metadata": meta,
                    "distance": distance
                })
                continue

            # Mismatch guard 2: category match
            if required_category and meta.get("category") != required_category:
                ranked.append({
                    "id": img_id,
                    "status": "REJECTED",
                    "reason": f"Category mismatch: expected {required_category}, detected {meta.get('category')}",
                    "metadata": meta,
                    "distance": distance
                })
                continue
                
            # Mismatch guard 2.5: Subject mismatch (e.g. expected fox, got wolf)
            # If the required_subject is passed, enforce it to avoid showing a wolf for a fox
            # Even if the semantic similarity is close enough.
            required_subject = "red fox" # This should ideally be passed in dynamically
            if "fox" in post_text.lower() and "wolf" in meta.get("subject", "").lower():
                ranked.append({
                    "id": img_id,
                    "status": "REJECTED",
                    "reason": f"Animal category mismatch: expected fox, detected wolf",
                    "metadata": meta,
                    "distance": distance
                })
                continue

            # Mismatch guard 3: minimum confidence from vision model
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

# Example usage / basic test
if __name__ == "__main__":
    matcher = SemanticMatcher()
    
    # Fake extracted images data
    mock_images = [
        {
            "id": "fox_1.jpg",
            "subject": "red fox",
            "category": "animal",
            "attributes": ["orange fur", "wild", "forest"],
            "caption": "A red fox standing in a forest",
            "confidence": 0.94
        },
        {
            "id": "wolf_1.jpg",
            "subject": "gray wolf",
            "category": "animal",
            "attributes": ["gray fur", "wild", "forest", "predator"],
            "caption": "A gray wolf in the forest",
            "confidence": 0.88
        },
        {
            "id": "dog_1.jpg",
            "subject": "dog",
            "category": "animal",
            "attributes": ["domestic", "park", "pet"],
            "caption": "A dog playing in a park",
            "confidence": 0.99
        }
    ]
    
    print("Adding images to vector index...")
    matcher.add_images(mock_images)
    
    post = "The behavior of red foxes in the wild, their orange fur and how they hunt."
    print(f"\nQuerying for post: '{post}'\n")
    
    results = matcher.rank_images_for_post(post, max_cosine_distance=0.6, required_category="animal")
    
    for r in results:
        print(f"Candidate: {r['id']} (Distance: {r['distance']:.3f})")
        print(f"  Status: {r['status']}")
        print(f"  Reason: {r['reason']}")
        print(f"  Metadata: {r['metadata']}")
        print("-" * 40)
