# Image Matching Engine

A robust, AI-powered background processing and semantic matching engine designed to pair blog posts with the perfect stock images.

## Core Features
1. **Asynchronous Vision AI Pipeline**: Users upload images instantly; a background worker uses Groq Vision AI to extract structured metadata (subject, category, attributes, caption).
2. **Semantic Matching**: Uses ChromaDB and dense vector embeddings (`all-MiniLM-L6-v2`) to match the conceptual meaning of a blog post to the visual contents of an image, going beyond keyword matching.
3. **Mismatch Guard**: A safety layer that automatically rejects bad recommendations based on high cosine distance, low vision confidence, or strict subject constraints.
4. **Review API**: A fully functional FastAPI backend with endpoints to queue images, check job status, rank images, and manually approve/reject pairings.
5. **Cost Tracking**: Per-call AI token costs are aggressively tracked and logged to `cost_log.jsonl`.

## Evaluation Metric
- **Top-1 Precision: 100.0%** (Tested on a labeled evaluation dataset of 14 complex queries).

## Architecture Diagram

```text
┌──────────────┐      POST /upload      ┌───────────────┐
│              ├───────────────────────►│               │
│  User / Web  │                        │  SQLite Queue │
│              │◄───────────────────────┤ (Pending Jobs)│
└──────────────┘     GET /status        └───────┬───────┘
                                                │
                                                ▼
┌─────────────────────────┐             ┌───────────────┐
│                         │   Extract   │               │
│ Vector DB (ChromaDB)    │◄────────────┤ Worker Daemon │
│ (Embeddings & Metadata) │             │ (Background)  │
│                         │             └───────┬───────┘
└──────────┬──────────────┘                     │
           │                                    ▼
           │                             ┌───────────────┐
           │ POST /rank                  │   Groq API    │
           │ (Mismatch Guard applied)    │ (Vision JSON) │
           ▼                             └───────────────┘
┌─────────────────────────┐
│                         │
│  Ranked & Safe Results  │
│                         │
└─────────────────────────┘
```

## How to Run

1. **Start the API Server**:
   ```bash
   python main.py
   ```
   Navigate to `http://localhost:8000/docs` to test endpoints.

2. **Start the Background Worker**:
   Open a separate terminal and run:
   ```bash
   python batch_processor.py --worker
   ```

3. **Run Automated Tests**:
   ```bash
   pytest test_system.py
   ```

4. **Run Top-1 Precision Eval**:
   ```bash
   python evaluate_precision.py
   ```
