from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import sqlite3
import os
import shutil

from batch_processor import BatchProcessor, DB_PATH, setup_db
from vector_db import VectorDB

app = FastAPI(title="Image Matching Engine API")

# Initialize shared components
processor = BatchProcessor()
vector_db = VectorDB()

# Ensure uploads directory exists
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Setup the review table
def setup_review_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_text TEXT,
            image_id TEXT,
            status TEXT,
            reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

setup_review_db()

# --- Phase 4: Background Processing API ---

@app.post("/images/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload a single image from your computer and queue it for background AI processing."""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Save the uploaded file to disk
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Enqueue the saved file path for the background worker
    processor.queue_image(file_path)
    
    return {
        "message": f"Successfully uploaded and queued {file.filename}.",
        "queued_file": file_path
    }

@app.get("/images/status")
def get_status():
    """Check the status of the background batch jobs."""
    return processor.get_progress()

# --- Phase 3 & 5: Matching and Review API ---

class RankRequest(BaseModel):
    post_text: str
    top_k: int = 3
    threshold: float = 0.65
    required_subject: Optional[str] = None

@app.post("/posts/rank")
def rank_images(req: RankRequest):
    """Rank images for a given post using semantic similarity and the Mismatch Guard."""
    candidates = vector_db.rank_images_for_post(
        post_text=req.post_text, 
        top_k=req.top_k, 
        max_cosine_distance=req.threshold,
        required_subject=req.required_subject
    )
    
    if not candidates:
        return {"message": "No confident match found. Similarity below threshold or no images available."}
        
    return {"candidates": candidates}

class ReviewRequest(BaseModel):
    post_text: str
    image_id: str
    status: str  # 'APPROVED' or 'REJECTED'
    reason: Optional[str] = None

@app.post("/review")
def review_match(req: ReviewRequest):
    """Approve or reject a suggested image-post pairing."""
    if req.status not in ["APPROVED", "REJECTED"]:
        raise HTTPException(status_code=400, detail="Status must be APPROVED or REJECTED")
        
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO reviews (post_text, image_id, status, reason) VALUES (?, ?, ?, ?)",
        (req.post_text, req.image_id, req.status, req.reason)
    )
    conn.commit()
    conn.close()
    
    return {"message": f"Successfully {req.status} the match."}

@app.get("/review/history")
def get_review_history():
    """Inspect the history of approvals and rejections."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM reviews ORDER BY created_at DESC LIMIT 50")
    rows = c.fetchall()
    conn.close()
    
    return {"reviews": [dict(r) for r in rows]}

if __name__ == "__main__":
    import uvicorn
    # To run: python main.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
