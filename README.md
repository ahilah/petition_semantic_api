# Municipal E-Petition Semantic Similarity & Deduplication API

<p align="center">
  <b>Master's Thesis Project | Lviv Polytechnic National University</b><br>
  Department of Automated Control Systems <br>
  <b>Author:</b> Anastasiia Hileta
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/FastAPI-REST%20API-teal.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/spaCy-NLP-green.svg" alt="spaCy">
  <img src="https://img.shields.io/badge/SQLite-Persistence-lightgrey.svg" alt="SQLite">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

---

## Overview

This project is developed as a Master's qualification thesis focusing on the automated detection of semantic similarity between electronic petitions and the identification of duplicate or closely related civic initiatives. 

The system implements a robust **Two-Stage Retrieval** architecture combining dense semantic vector embeddings (`intfloat/multilingual-e5-small`) with a custom domain-specific rule engine (`DomainContextEngine`), continuous geospatial distance validation (Haversine formula), relational persistence (SQLite), and unsupervised machine learning modules for municipal governance platforms.

---

## Key Features

- **Semantic Deduplication:** Vector-based similarity search across active municipal petition corpora using PyTorch and Sentence-Transformers.
- **Batch Processing API:** Mass validation endpoint for simultaneous processing of administrative petition batches.
- **Unsupervised Semantic Clustering:** K-Means clustering (`scikit-learn`) over vector embeddings to uncover latent civic topics and trends in real-time.
- **Active Learning (Human-in-the-Loop):** Expert feedback loop allowing municipal moderators to confirm duplicates and build a verified Ground Truth corpus.
- **Geospatial GeoJSON Export:** Spatial data structuring for instant integration with GIS mapping tools (Leaflet/Folium).
- **SQLite Audit & Telemetry:** Built-in relational logging of system performance, query metrics, latency, and verification verdicts.
- **Dual-Channel Logging:** Comprehensive event tracking into both console streams and file logs (`app.log`).
- **Automated Audit Reporting:** Generation of official review protocols and audit reports in CSV and PDF formats (`fpdf2`).

---

## Technology Stack

- **Backend Framework:** FastAPI, Uvicorn, Pydantic
- **NLP & Embeddings:** Sentence-Transformers (`intfloat/multilingual-e5-small`), spaCy (`uk_core_news_lg`)
- **Machine Learning & Analytics:** PyTorch, Scikit-learn (K-Means), Pandas, NumPy
- **Database & Storage:** SQLite (Telemetry & Feedback persistence)
- **Geospatial & Reporting:** Haversine formula, GeoJSON, FPDF2 (PDF generation)

---

## Project Structure

```text
petition_semantic_api/
│
├── assets/                 # Graphical resources and fonts for PDF reports
│   └── DejaVuSans.ttf
│
├── data/                   # Corpora, serialized embeddings, and SQLite database
│   ├── petitions_cleaned.csv
│   ├── embeddings.pt
│   └── petition_analytics.db
│
├── main.py                 # FastAPI application and core API endpoints
├── database.py             # SQLite persistence, telemetry, and analytics queries
├── matcher.py              # Two-stage retrieval and semantic search pipeline
├── engine.py               # Domain context engine, NER rules, and entity extraction
├── geocoding.py            # Geocoding utilities and Haversine distance calculations
├── reports.py              # CSV and PDF audit protocol generator
├── validation.py           # Input text validation and anti-spam logic
├── Dockerfile              # Containerization configuration
├── docker-compose.yml      # Multi-container orchestration setup
└── requirements.txt        # Project dependencies
```


## Installation & Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/petition_semantic_api.git](https://github.com/your-username/petition_semantic_api.git)
cd petition_semantic_api
```

---

### 2. Create a Virtual Environment

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download spaCy Ukrainian Language Model

```bash
python -m spacy download uk_core_news_lg
```

---

## Running the Application

To launch the interactive municipal monitoring web interface, run:

```bash
uvicorn main:app --reload
```

Open your browser and navigate to the interactive Swagger UI documentation:
```text
http://localhost:8000/docs
```

---