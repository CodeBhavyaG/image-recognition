import json
from vector_db import VectorDB

def main():
    db = VectorDB()

    with open("posts.json", "r") as f:
        posts = json.load(f)

    all_rankings = []

    for post in posts:
        # Get the top 2 candidate images
        candidates = db.rank_images_for_post(post["text"], top_k=2, max_cosine_distance=0.65)

        # Format the output for easy review
        post_result = {
            "post_id": post["id"],
            "post_text": post["text"],
            "matches": []
        }

        for cand in candidates:
            post_result["matches"].append({
                "image_id": cand["id"],
                "status": cand["status"],
                "reason": cand["reason"],
                "distance": float(cand["distance"]), # Convert to standard float for JSON
                "subject": cand["metadata"].get("subject")
            })

        all_rankings.append(post_result)

    with open("rankings.json", "w") as f:
        json.dump(all_rankings, f, indent=2)

    print("Ranked images for 50 posts and saved to rankings.json!")

if __name__ == "__main__":
    main()
