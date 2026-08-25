import sqlite3
import time
import json
from pathlib import Path
from typing import List, Dict, Any

from groq_extract import extract_understanding, get_client
from vector_db import VectorDB

DB_PATH = "batch_jobs.sqlite"

def setup_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT UNIQUE,
            status TEXT DEFAULT 'PENDING',
            retries INTEGER DEFAULT 0,
            error_msg TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

class BatchProcessor:
    def __init__(self, max_retries: int = 3):
        setup_db()
        self.max_retries = max_retries
        self.client = get_client()
        self.vector_db = VectorDB()

    def queue_image(self, image_path: str):
        """Enqueue an image for background processing. Returns immediately."""
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        try:
            c.execute(
                "INSERT INTO jobs (image_path, status) VALUES (?, 'PENDING')",
                (image_path,)
            )
            conn.commit()
            print(f"Queued: {image_path}")
        except sqlite3.IntegrityError:
            print(f"Already in queue: {image_path}")
        finally:
            conn.close()

    def queue_images(self, image_paths: List[str]):
        """Enqueue multiple images at once."""
        for path in image_paths:
            self.queue_image(path)

    def get_progress(self) -> Dict[str, int]:
        """Returns the current progress of the batch jobs."""
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status")
        rows = c.fetchall()
        conn.close()
        return {row[0]: row[1] for row in rows}

    def process_queue(self):
        """The background worker that processes PENDING jobs."""
        print("Starting background batch worker...")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        while True:
            # Find a pending job
            c.execute("SELECT id, image_path, retries FROM jobs WHERE status = 'PENDING' LIMIT 1")
            row = c.fetchone()
            
            if not row:
                print("Queue is empty. Worker sleeping for 5 seconds...")
                time.sleep(5)
                continue

            job_id, image_path, retries = row
            
            # Mark as processing
            c.execute("UPDATE jobs SET status = 'PROCESSING', updated_at = CURRENT_TIMESTAMP WHERE id = ?", (job_id,))
            conn.commit()

            print(f"Processing job {job_id} for {image_path}...")
            
            try:
                # 1. Vision API Call (with cost tracking handled inside groq_extract.py)
                understanding = extract_understanding(image_path, client=self.client)
                
                # 2. Vector DB Embedding & Storage (Local, free)
                self.vector_db.add_image_understanding(image_id=image_path, understanding=understanding)
                
                # Mark as completed
                c.execute("UPDATE jobs SET status = 'COMPLETED', updated_at = CURRENT_TIMESTAMP WHERE id = ?", (job_id,))
                conn.commit()
                print(f"Job {job_id} COMPLETED.")
                
            except Exception as e:
                error_msg = str(e)
                new_retries = retries + 1
                
                if new_retries >= self.max_retries:
                    status = 'FAILED'
                else:
                    status = 'PENDING'  # Put back in queue to retry
                
                c.execute(
                    "UPDATE jobs SET status = ?, retries = ?, error_msg = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (status, new_retries, error_msg, job_id)
                )
                conn.commit()
                print(f"Job {job_id} {status}. Attempt {new_retries}/{self.max_retries}. Error: {error_msg}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Background Batch Processing System")
    ap.add_argument("--queue", nargs="+", help="Queue image(s) for processing")
    ap.add_argument("--worker", action="store_true", help="Start the background worker process")
    ap.add_argument("--status", action="store_true", help="Check progress of the queue")
    args = ap.parse_args()

    processor = BatchProcessor()

    if args.queue:
        processor.queue_images(args.queue)
    elif args.status:
        progress = processor.get_progress()
        print("\nBatch Processing Progress:")
        for status, count in progress.items():
            print(f"  {status}: {count}")
    elif args.worker:
        processor.process_queue()
    else:
        ap.print_help()
