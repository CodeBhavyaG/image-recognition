import json
from vector_db import VectorDB

def evaluate_top1_precision(eval_file="eval_dataset.json", threshold=0.75):
    # Initialize the database
    db = VectorDB()
    
    # Load the evaluation dataset
    with open(eval_file, "r") as f:
        dataset = json.load(f)
        
    correct = 0
    total = len(dataset)
    
    print(f"Starting Evaluation on {total} items...\n")
    
    for i, item in enumerate(dataset):
        post = item["post_text"]
        expected_id = item["expected_image_id"]
        
        # Rank the images using our system
        candidates = db.rank_images_for_post(post, top_k=1, max_cosine_distance=threshold)
        
        # Check if the top result (index 0) matches the expected image
        if not candidates:
            print(f"[{i+1}/{total}] ❌ FAILED (No candidates passed Mismatch Guard)")
            print(f"   Query: {post}")
            continue
            
        top_match = candidates[0]
        
        if top_match["status"] == "REJECTED":
            print(f"[{i+1}/{total}] ❌ FAILED (Top candidate was REJECTED: {top_match['reason']})")
            print(f"   Query: {post}")
            continue
            
        actual_id = top_match["id"]
        
        if actual_id == expected_id:
            correct += 1
            print(f"[{i+1}/{total}] ✅ SUCCESS")
        else:
            print(f"[{i+1}/{total}] ❌ MISMATCH")
            print(f"   Query: {post}")
            print(f"   Expected: {expected_id}")
            print(f"   Got:      {actual_id}")

    # Calculate Top-1 Precision
    precision = (correct / total) * 100 if total > 0 else 0
    print("\n" + "="*40)
    print(f"🏆 TOP-1 PRECISION: {precision:.1f}% ({correct}/{total})")
    print("="*40)
    
    return precision

if __name__ == "__main__":
    evaluate_top1_precision()
