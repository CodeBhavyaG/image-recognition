# Capstone Evidence

This file serves as proof of completion for the AI Image Understanding & Content Matching Engine capstone project.

### 1. Vision AI & Schema Validation
* Built in `groq_extract.py`. Uses `Pydantic` to enforce a strict `ImageUnderstanding` schema (`subject`, `category`, `attributes`, `caption`, `confidence`). Rejects malformed AI output via a 3-strike retry loop.

### 2. Cost Tracking
* Captured via `groq_extract.py`. Every successful vision call extracts `prompt_tokens` and `completion_tokens` and appends them to a persistent `cost_log.jsonl` file.

### 3. Background Job Processing
* Built in `batch_processor.py`. Uses a persistent SQLite table (`jobs`) to model FlyRank's async classification pattern. 
* Prevents the FastAPI server from hanging during 10-second vision calls. Provides progress tracking (`--status`) and retries.

### 4. Semantic Matching & Vector DB
* Built in `vector_db.py`. Uses local `ChromaDB` with `all-MiniLM-L6-v2` dense embeddings.
* Proved capability to match exact concepts without exact keywords (e.g., matching a post about "how a cow walks" specifically to a walking chicken rather than a standing bull, before strict subject guards are applied).

### 5. Safety Layer / Mismatch Guard
* Integrated into `vector_db.rank_images_for_post()`.
* Automatically rejects images if:
  1. Cosine distance exceeds the strict threshold (default `0.65`).
  2. Vision model confidence is below `0.70`.
  3. The `required_subject` provided by the API does not match the image's subject tag.
* Yields human-readable explanations (e.g., `"Subject mismatch: expected cow, detected chicken"`).

### 6. API Backend & Review Workflow
* Built in `main.py` using `FastAPI`.
* Exposes `POST /images/upload` (for queueing files to the background worker), `GET /images/status` (progress tracking), `POST /posts/rank` (Semantic ranking), and `POST /review` (to record human approvals/rejections in SQLite).

### 7. Quality Assurance
* **Automated Tests**: `test_system.py` uses `pytest` to guarantee schema validation and mismatch guard enforcement.
* **Eval Metric**: `evaluate_precision.py` runs against `eval_dataset.json` mapping 14 blog posts to 14 expected images. The system scored **100.0% Top-1 Precision**.
