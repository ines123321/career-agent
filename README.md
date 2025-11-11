
# Career Agent - Optimisation Intelligente de CV

##  Description

**Career Agent** est une application web qui utilise l'IA pour analyser et optimiser automatiquement votre CV en fonction d'offres d'emploi pertinentes. Le système combine LLM, scraping et génération de documents pour créer des CV personnalisés.

##  Fonctionnalités Clés

- ** Analyse de CV** : Extraction automatique depuis PDF avec Groq (Llama 3.3 70B)
- ** Matching Intelligent** : Recherche d'offres LinkedIn + scoring contextuel avancé
- ** Enhancement Éthique** : Reformulation contextuelle avec validation RAG
- ** Génération PDF** : Création automatique de CV professionnels en LaTeX

##  Architecture

```python
PDF → Extraction → Analyse → Matching → Enhancement → LaTeX → PDF
```

### Composants Principaux
- **PDFCVParser** : Extraction et parsing des CV
- **JobScraper** : Scraping d'offres LinkedIn (Apify)
- **JobMatcher** : Matching contextuel avec Groq
- **CVEnhancer** : Optimisation éthique avec RAG
- **LaTeXGenerator** : Génération de CV professionnels

##  Installation Rapide

```bash
# 1. Cloner le projet
git clone https://github.com/ines123321/career-agent.git
cd career-agent

# 2. Environnement virtuel
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Dépendances
pip install -r requirements.txt

# 4. Configuration
cp .env.example .env
# Éditer .env avec vos clés API
```

##  Configuration

Fichier `.env` requis :
```env
GROQ_API_KEY=your_groq_key
APIFY_API_KEY=your_apify_key
FLASK_SECRET_KEY=your_secret_key
```

**Prérequis** : Python 3.8+, pdflatex, 4GB+ RAM

##  Utilisation

```bash
python app.py
```
Ouvrir `http://localhost:5000`

### Workflow
1.  Uploader un CV PDF
2.  Analyse automatique
3.  Recherche d'offres pertinentes
4.  Matching intelligent
5.  Optimisation contextuelle
6.  Génération du CV optimisé

##  API Principale

- `POST /api/process-cv` : Traitement complet
- `GET /api/download/<file>` : Téléchargement
- `POST /api/regenerate-enhancement` : Régénération

##  Technologies

- **Backend** : Flask, Groq API, Apify
- **AI** : Llama 3.3 70B, Sentence Transformers, RAG
- **Document** : LaTeX, PyPDF2, pdfplumber
- **Orchestration** : LangGraph

##  Dépannage

**Erreur LaTeX** : Vérifier l'installation de pdflatex
**Scraping échoué** : Vérifier la clé Apify et les quotas
**Enhancement échoué** : Vérifier la clé Groq et la structure du CV

