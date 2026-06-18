# ClaimShield AI - FastAPI Backend Engine 🛡️

The backend engine of **ClaimShield AI** is a robust, concurrent, and secure FastAPI server that orchestrates the claims auditing and document ingestion pipeline. Powered by **LangGraph** agents, **Hybrid RAG**, and **Machine Learning anomaly detection**, it automates claims evaluation, verifies compliance, and calculates fraud indexes in seconds.

---

## ⚙️ Technical Architecture & Features

### 1. 🤖 LangGraph Stateful Pipeline
The claims auditing workflow is modeled as a state machine (`StateGraph`) that flows through six specialized agent nodes:
- **Ingest Agent:** Validates documents, runs OCR text extraction, and identifies PII.
- **Policy Agent:** Extracts keywords and queries Qdrant Vector indexes for matched clauses, trust registries, and historical statistics.
- **Claim Agent:** Evaluates medical codes, treatment matches, and deductible policies.
- **Fraud Agent:** Passes vectorized claims features to the XGBoost classifier and Isolation Forest outlier detector.
- **Report Agent:** Compiles the findings into an structured Markdown dossier.
- **Decision Agent:** Synthesizes final recommendations (Approve, Deny, Investigate) and confidence ratings.

### 2. 📄 GPT-4o Vision OCR & PII Scrubbing
- Scans claim invoices and incident reports using **GPT-4o Vision** for high-precision OCR extraction of tables and pricing.
- Integrates **Microsoft Presidio** to detect and mask Personally Identifiable Information (PII) like names, SSNs, and phone numbers before interacting with cloud LLMs, ensuring HIPAA and GDPR compliance.

### 3. 🔍 Hybrid RAG & Vector Search
- Policies and trust registries are chunked and indexed into a **Qdrant Vector Database**.
- Uses hybrid retrieval (0.6 cosine similarity + 0.4 BM25 keyword score) to retrieve document clauses.
- Retains database logs and matched policy references for decision transparency.

### 4. 🚨 Fraud Analytics Engine
- Combines predictive machine learning with deterministic heuristics:
  - **XGBoost:** Evaluates historical features to identify known fraud patterns.
  - **Isolation Forest:** Flag statistical outliers (unusual cost ratios, weekend claims, duplicate invoice providers).
- Integrates with **MLflow** to track real-time runs and log performance parameters.

### 5. 🗄️ Storage & Dynamic Streaming
- Saves claim documents securely on Cloudflare R2 S3-compatible storage.
- Features a secure, authenticated `/claims/download/{file_path}` streaming route that downloads PDFs/images from S3 directly to the user.
- Employs fallback routing to local directories (`./data`) if Cloudflare R2 is offline.

---

## 📂 Project Structure

```text
backend/
├── app/
│   ├── api/          # Route Controllers (auth, claims, policy, feedback)
│   ├── core/         # Settings (Pydantic), security, telemetry (OTel)
│   ├── db/           # SQL Session and SQLAlchemy Models
│   ├── schemas/      # Pydantic Schemas for data validation
│   ├── services/     # Services (OCR, PII analyzer, Email, LLM)
│   ├── rag/          # Qdrant Vector store indices, chunkers, retrievers
│   ├── agents/       # LangGraph Agent Nodes (Ingest, Policy, Fraud, etc.)
│   ├── workers/      # Celery task queue configurations
│   └── main.py       # FastAPI application initialisation
├── data/             # Local storage fallback directory
├── monitoring/       # Prometheus metric configurations
├── tests/            # Test suite (Pytest)
├── requirements.txt  # Python requirements
└── .env              # Environment config file
```

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10** or higher
- **Poetry** or **Venv**
- **Qdrant Cloud/Local instance** (Vector DB)
- **MinIO or Cloudflare R2 account** (S3 storage)
- **MLflow Tracking server** (Optional)

### Installation & Run

1. Navigate to the backend directory and set up a virtual environment:
   ```bash
   cd backend
   python -m venv .venv
   ```

2. Activate the virtual environment:
   - **Windows:**
     ```bash
     .venv\Scripts\activate
     ```
   - **Mac/Linux:**
     ```bash
     source .venv/bin/activate
     ```

3. Install python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables (`.env`):
   Create a `.env` file in the `backend` folder and populate it with:
   ```env
   SECRET_KEY=your-jwt-secret-key
   OPENAI_API_KEY=your-openai-api-key
   SUPABASE_URL=your-supabase-url
   DATABASE_URL=postgresql+asyncpg://...
   REDIS_URL=redis://localhost:6379
   BACKEND_URL=http://localhost:8000
   FRONTEND_URL=http://localhost:5173
   MINIO_ENDPOINT=your-account-id.r2.cloudflarestorage.com
   MINIO_ACCESS_KEY=your-access-key
   MINIO_SECRET_KEY=your-secret-key
   QDRANT_URL=your-qdrant-url
   QDRANT_API_KEY=your-qdrant-api-key
   MLFLOW_TRACKING_URI=http://localhost:5000
   ```

5. Run the server:
   ```bash
   uvicorn app.main:app --reload
   ```
   The interactive Swagger documentation will be available at `http://localhost:8000/docs`.

### Running Tests
Execute the test suite to verify code integrity:
```bash
python -m pytest
```

---

*Asynchronous, stateless, and state-of-the-art backend auditing processor.*
