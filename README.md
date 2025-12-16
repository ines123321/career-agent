# Career Agent - Intelligent Resume Optimization

## Description

**Career Agent** is a web application that uses AI to automatically analyze and optimize your resume based on relevant job offers. The system combines LLM, scraping, and document generation to create personalized resumes.

## Key Features

- **Resume Analysis**: Automatic extraction from PDF with Groq (Llama 3.3 70B)
- **Intelligent Matching**: LinkedIn job search + advanced contextual scoring
- **Ethical Enhancement**: Contextual reformulation with RAG validation
- **PDF Generation**: Automatic creation of professional resumes in LaTeX

## Architecture

```python
PDF → Extraction → Analysis → Matching → Enhancement → LaTeX → PDF
```

### Main Components

- **PDFCVParser**: Resume extraction and parsing
- **JobScraper**: LinkedIn job scraping (Apify)
- **JobMatcher**: Contextual matching with Groq
- **CVEnhancer**: Ethical optimization with RAG
- **LaTeXGenerator**: Professional resume generation

## Quick Installation

```bash
# 1. Clone the project
git clone https://github.com/ines123321/career-agent.git
cd career-agent

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Dependencies
pip install -r requirements.txt

# 4. Configuration
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Required `.env` file:

```env
GROQ_API_KEY=your_groq_key
APIFY_API_KEY=your_apify_key
FLASK_SECRET_KEY=your_secret_key
```

**Prerequisites**: Python 3.8+, pdflatex, 4GB+ RAM

## Usage

```bash
python app.py
```

Open `http://localhost:5000`

### Workflow

1.  Upload a PDF resume
2.  Automatic analysis
3.  Search for relevant job offers
4.  Intelligent matching
5.  Contextual optimization
6.  Generate optimized resume

## Main API

- `POST /api/process-cv`: Complete processing
- `GET /api/download/<file>`: Download
- `POST /api/regenerate-enhancement`: Regeneration

## Technologies

- **Backend**: Flask, Groq API, Apify
- **AI**: Llama 3.3 70B, Sentence Transformers, RAG
- **Document**: LaTeX, PyPDF2, pdfplumber
- **Orchestration**: LangGraph

## Troubleshooting

**LaTeX Error**: Check pdflatex installation

**Scraping Failed**: Check Apify key and quotas

**Enhancement Failed**: Check Groq key and resume structure
