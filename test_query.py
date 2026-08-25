import argparse
from vector_db import VectorDB
import json

def main():
    ap = argparse.ArgumentParser(description="Test querying the Vector DB.")
    ap.add_argument("--post", required=True, help="The blog post text to match images against.")
    ap.add_argument("--topk", type=int, default=3, help="Number of images to retrieve.")
    ap.add_argument("--threshold", type=float, default=0.6, help="Cosine distance threshold.")
    args = ap.parse_args()

    db = VectorDB()
    
    print(f"\nQuerying vector DB for post:\n'{args.post}'\n")
    results = db.rank_images_for_post(args.post, top_k=args.topk, max_cosine_distance=args.threshold)
    
    if not results:
        print("No candidates found in DB.")
        return

    for idx, res in enumerate(results, start=1):
        print(f"--- Candidate {idx} ---")
        print(f"Image ID: {res['id']}")
        print(f"Distance: {res['distance']:.3f} (Lower is better)")
        print(f"Status:   {res['status']}")
        if res['reason'] != "Good match":
            print(f"Reason:   {res['reason']}")
        print(f"Metadata: {json.dumps(res['metadata'], indent=2)}\n")

if __name__ == "__main__":
    main()
