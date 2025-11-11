import os
import uuid
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, session
from werkzeug.utils import secure_filename
import sys
import io
import contextlib
import re
import ast
import io
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from groq import Groq
from apify_client import ApifyClient
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from urllib.parse import quote
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import pdfplumber
import subprocess
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
from dotenv import load_dotenv




# Configuration
UPLOAD_FOLDER = 'uploads'
TEMP_FOLDER = 'temp'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMP_FOLDER'] = TEMP_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Créer les dossiers nécessaires
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Votre code principal intégré ici
# ================================
# COLLER TOUT VOTRE CODE EXISTANT ICI 
load_dotenv() 

class CareerAgentConfig:
    def __init__(self):
        self.APIFY_API_KEY = os.getenv("APIFY_API_KEY")
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.max_jobs_to_fetch = 5
        self.similarity_threshold = 0.4

config = CareerAgentConfig()



class GroqProcessor:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.model_name = "llama-3.3-70b-versatile"
        print(f"✅ Groq Processor initialisé avec {self.model_name}")

    def generate_text(self, prompt, max_tokens=1024, temperature=0.7):
        """Génère du texte avec Groq API"""
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un assistant IA expert en analyse de CV, offres d'emploi et recrutement. Tu fournis des analyses contextuelles précises et des recommandations professionnelles."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                stream=False
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"❌ Erreur Groq API: {e}")
            return f"Erreur de génération: {str(e)}"




class PDFCVParser:
    def __init__(self, groq_processor):
        self.groq = groq_processor
        
    def extract_text_from_pdf(self, pdf_path):
        """Extrait le texte brut du PDF"""
        text = ""
        try:
            # Méthode 1: pdfplumber (meilleure pour les tableaux)
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            print(f"✅ Texte extrait du PDF ({len(text)} caractères)")
        except Exception as e:
            print(f"⚠️ Erreur pdfplumber, essai PyPDF2: {e}")
            # Fallback: PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            except Exception as e2:
                print(f"❌ Erreur extraction PDF: {e2}")
                return None
        
        return text.strip()
    
    def parse_cv_with_llm(self, pdf_text):
        """Parse le CV avec Groq pour extraire les informations structurées"""
        
        # Détecter la langue
        language = self._detect_language(pdf_text)
        
        if language == "french":
            prompt = self._build_french_parsing_prompt(pdf_text)
        else:
            prompt = self._build_english_parsing_prompt(pdf_text)
        
        try:
            response = self.groq.generate_text(prompt, max_tokens=1500)
            
            # Parser le JSON
            cv_data = self._parse_json_response(response)
            
            if cv_data:
                print("✅ CV parsé avec succès par Groq")
                return cv_data
            else:
                print("❌ Échec du parsing JSON")
                return None
                
        except Exception as e:
            print(f"❌ Erreur parsing avec Groq: {e}")
            return None
    
    def _detect_language(self, text):
        """Détecte la langue du CV"""
        french_keywords = ['compétences', 'expérience', 'formation', 'profil', 'diplôme']
        english_keywords = ['skills', 'experience', 'education', 'profile', 'degree']
        
        text_lower = text.lower()
        french_count = sum(1 for kw in french_keywords if kw in text_lower)
        english_count = sum(1 for kw in english_keywords if kw in text_lower)
        
        return "french" if french_count > english_count else "english"
    
    def _build_french_parsing_prompt(self, pdf_text):
        """Prompt en français pour parser le CV"""
        return f"""
Tu es un expert en analyse de CV. Extrait TOUTES les informations de ce CV au format JSON STRICT.

CV:
{pdf_text[:4000]}

FORMAT JSON OBLIGATOIRE:
{{
    "nom_complet": "nom et prénom",
    "email": "email",
    "telephone": "téléphone",
    "job_title": "poste actuel ou recherché",
    "localisation": "ville, pays",
    "competences_techniques": ["liste des compétences techniques"],
    "competences_soft": ["liste des soft skills"],
    "experiences": ["expérience 1", "expérience 2", ...],
    
    "formation": "diplôme principal",
    "objectif_professionnel": "résumé/profil/objectif"
}}

RÈGLES:
- Extrait TOUT ce qui est présent
- Les sections peuvent s'appeler: Profil/Summary, Projets/Projects, Formation/Education
- Accepte les variantes françaises ET anglaises
- Pour "localisation", indique le PAYS (France, Canada, etc.)
- Si une info manque, mets ""
- Retourne UNIQUEMENT le JSON

JSON:
"""
    
    def _build_english_parsing_prompt(self, pdf_text):
        """Prompt en anglais pour parser le CV"""
        return f"""
You are a CV parsing expert. Extract ALL information from this CV in STRICT JSON format.

CV:
{pdf_text[:4000]}

REQUIRED JSON FORMAT:
{{
    "nom_complet": "full name",
    "email": "email",
    "telephone": "phone",
    "job_title": "current or target job title",
    "localisation": "city, country",
    "competences_techniques": ["technical skills list"],
    "competences_soft": ["soft skills list"],
    "experiences": ["experience 1", "experience 2", ...],
    
    "formation": "main degree",
    "objectif_professionnel": "summary/profile/objective"
}}

RULES:
- Extract EVERYTHING present
- Sections may be called: Profile/Summary, Projects, Education/Formation
- Accept both French and English variants
- For "localisation", specify the COUNTRY (France, Canada, etc.)
- If info missing, use ""
- Return ONLY JSON

JSON:
"""
    
    def _parse_json_response(self, response):
     """Parse la réponse JSON - VERSION AVEC NETTOYAGE DES STRINGS"""
     try:
         response = response.strip()
         response = re.sub(r'```json\s*', '', response)
         response = re.sub(r'```\s*', '', response)
         response = re.sub(r'^[^{]*', '', response)  # Supprimer tout avant {
         response = re.sub(r'[^}]*$', '', response)  # Supprimer tout après }
        
         json_match = re.search(r'\{.*\}', response, re.DOTALL)
         if json_match:

            
             cv_data = json.loads(json_match.group(0))
            
            # 🔥 CORRECTION CRITIQUE: Nettoyer les expériences
             if 'experiences' in cv_data and isinstance(cv_data['experiences'], list):
                 cleaned_experiences = []
                 for exp in cv_data['experiences']:
                     if isinstance(exp, str):
                        # Si c'est une string qui ressemble à un dict Python
                         if exp.strip().startswith('{'):
                             try:
                                # Convertir la string en dict Python
                                 exp_dict = ast.literal_eval(exp)
                                # Extraire les tâches et formatter proprement
                                 entreprise = exp_dict.get('entreprise', '')
                                 poste = exp_dict.get('poste', '')
                                 date_debut = exp_dict.get('date_debut', '')
                                 date_fin = exp_dict.get('date_fin', '')
                                 taches = exp_dict.get('taches', [])
                                
                                # Créer une description propre
                                 exp_text = f"{entreprise} - {poste} ({date_debut} - {date_fin})"
                                 if taches:
                                     exp_text += ": " + " • ".join(taches[:3])  # Max 3 tâches
                                
                                 cleaned_experiences.append(exp_text)
                             except:
                                # Si échec, garder la string telle quelle
                                 cleaned_experiences.append(exp)
                         else:
                            # C'est déjà une string normale
                             cleaned_experiences.append(exp)
                     else:
                         cleaned_experiences.append(str(exp))
                
                 cv_data['experiences'] = cleaned_experiences
            
            # 🔥 CORRECTION: Nettoyer la formation
             if 'formation' in cv_data and isinstance(cv_data['formation'], str):
                 formation_str = cv_data['formation']
                 if formation_str.strip().startswith('{'):
                     try:
                        # Convertir la string en dict
                         formation_dict = ast.literal_eval(formation_str)
                         diplome = formation_dict.get('diplome', '')
                         etablissement = formation_dict.get('etablissement', '')
                        
                        # Créer un string propre
                         if diplome and etablissement:
                             cv_data['formation'] = f"{diplome}, {etablissement}"
                         elif diplome:
                             cv_data['formation'] = diplome
                         else:
                             cv_data['formation'] = formation_str
                     except:
                        # Garder tel quel si échec
                         pass
            
            # Validation des champs requis
             required_fields = ['nom_complet', 'job_title', 'localisation',
                              'competences_techniques', 'experiences']
             if all(field in cv_data for field in required_fields):
                 return cv_data
        
         return None
     except Exception as e:
         print(f"⚠️ Erreur parsing JSON: {e}")
         return None
    
    def process_pdf_cv(self, pdf_path):
        """Pipeline complet: PDF → Texte → JSON structuré"""
        print(f"📄 Traitement du PDF: {pdf_path}")
        
        # Étape 1: Extraction texte
        pdf_text = self.extract_text_from_pdf(pdf_path)
        if not pdf_text:
            return None
        
        # Étape 2: Parsing avec LLM
        cv_data = self.parse_cv_with_llm(pdf_text)
        if not cv_data:
            return None
        
        # Ajouter le texte original pour référence
        cv_data['text_original'] = pdf_text[:3000]
        
        print(f"✅ CV structuré extrait:")
        print(f"   👤 {cv_data.get('nom_complet')}")
        print(f"   🎯 {cv_data.get('job_title')}")
        print(f"   📍 {cv_data.get('localisation')}")
        print(f"   🔧 {len(cv_data.get('competences_techniques', []))} compétences")
        
        return cv_data
# ============================================================
# PROFILE ANALYZER HYBRIDE (Groq + données structurées)
# ============================================================

class ProfileAnalyzer:
    def __init__(self, groq_processor):
        self.groq = groq_processor

    def analyze_cv(self, cv_data):
        """Analyse hybride : données structurées + enrichment contextuel Groq"""
        print("🧠 Analyse hybride du CV...")

        # ÉTAPE 1: Utiliser les données structurées EXISTANTES (CRITIQUE)
        profile = {
            "nom_complet": cv_data.get('nom_complet', ''),
            "email": cv_data.get('email', ''),
            "telephone": cv_data.get('telephone', ''),
            "job_title": cv_data.get('job_title', ''),
            "localisation": cv_data.get('localisation', ''),
            "competences_techniques": cv_data.get('competences_techniques', []),
            "competences_soft": cv_data.get('competences_soft', []),
            "experiences": cv_data.get('experiences', []),
            "projets": cv_data.get('projets', []),
            "formation": cv_data.get('formation', ''),
            "objectif_professionnel": cv_data.get('objectif_professionnel', '')
        }

        # VALIDATION CRITIQUE (comme dans l'ancien code)
        if not profile['localisation']:
            raise Exception("Localisation manquante dans le CV")
        if not profile['job_title']:
            raise Exception("Job title manquant dans le CV")


        # Validation finale
        profile = self._validate_profile(profile)
        profile['raw_cv_text'] = self._prepare_cv_text(cv_data)[:2000]

        print("🎯 PROFIL ANALYSÉ:")
        print(f"👤 {profile['nom_complet']}")
        print(f"🎯 {profile['job_title']}")
        print(f"📍 {profile['localisation']}")  # ← MAINTENANT ça affichera "France"
        print(f"🔧 {len(profile['competences_techniques'])} compétences techniques")
        print(f"📁 {len(profile['projets'])} projets")
        print("="*50)

        return profile

    def _enrich_with_groq(self, base_profile, cv_data):
        """Enrichit le profil avec des insights contextuels Groq"""
        cv_text = self._prepare_cv_text(cv_data)

        prompt = f"""
        Tu es un expert en recrutement. Tu dois ENRICHIR le profil suivant avec des insights contextuels.

        PROFIL DE BASE (déjà structuré):
        {json.dumps(base_profile, indent=2, ensure_ascii=False)}

        TEXTE COMPLET DU CV (pour contexte):
        {cv_text[:2000]}

        TÂCHE: Apporte des améliorations contextuelles subtiles:
        1. Affiner le job_title si nécessaire (ex: "Data Scientist" → "Data Scientist Senior")
        2. Identifier des compétences techniques IMPLIQUÉES mais non explicites
        3. Reformuler l'objectif professionnel pour plus d'impact
        4. Suggérer des compétences soft supplémentaires pertinentes

        IMPORTANT:
        - NE PAS changer la localisation, email, téléphone, nom
        - NE PAS inventer des compétences non plausibles
        - GARDER la structure JSON existante

        Retourne UNIQUEMENT les champs améliorés (pas besoin de tout répéter).
        """

        try:
            response = self.groq.generate_text(prompt, max_tokens=800)

            # Extraction du JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                enriched_data = json.loads(json_match.group(0))
                print("✅ Enrichissement contextuel Groq réussi")
                return enriched_data
            else:
                print("⚠️ Aucun JSON d'enrichissement trouvé")
                return {}

        except Exception as e:
            print(f"❌ Erreur enrichissement Groq: {e}")
            return {}

    def _prepare_cv_text(self, cv_data):
        """Prépare le texte du CV pour l'analyse contextuelle - CORRIGÉ"""
        components = []

        # Texte original s'il existe
        if cv_data.get('text_original'):
            components.append(str(cv_data['text_original']))

        # Ajouter les champs structurés de manière sécurisée
        fields_to_include = ['nom_complet', 'job_title', 'competences_techniques',
                           'competences_soft', 'experiences','projets','formation',
                           'objectif_professionnel', 'localisation']

        for field in fields_to_include:
            if cv_data.get(field):
                value = cv_data[field]
                # Gérer les listes
                if isinstance(value, list):
                    # Filtrer les éléments None et convertir en string
                    clean_list = [str(item) for item in value if item is not None]
                    if clean_list:
                        components.append(f"{field}: {', '.join(clean_list)}")
                # Gérer les dictionnaires
                elif isinstance(value, dict):
                    # Convertir le dict en string lisible
                    dict_str = ', '.join([f"{k}: {v}" for k, v in value.items() if v])
                    if dict_str:
                        components.append(f"{field}: {dict_str}")
                # Gérer les chaînes et autres types
                else:
                    value_str = str(value).strip()
                    if value_str:
                        components.append(f"{field}: {value_str}")

        return "\n".join(components)[:3000]

    def _validate_profile(self, profile):
        """Valide et nettoie le profil"""
        default_profile = {
            "competences_techniques": [],
            "competences_soft": [],
            "experiences": [],
            "projets": [],
            "formation": "",
            "objectif_professionnel": "",
            "localisation": "",
            "nom_complet": "",
            "telephone": "",
            "email": "",
            "job_title": ""
        }

        for key, default in default_profile.items():
            if key not in profile:
                profile[key] = default
            elif profile[key] is None:
                profile[key] = default

        # Nettoyer les listes - CORRIGÉ
        def clean_list(items):
            """Nettoie une liste en gérant les différents types"""
            if not isinstance(items, list):
                return []
            cleaned = []
            for item in items:
                if item is None:
                    continue
                # Convertir en string si nécessaire
                item_str = str(item).strip() if not isinstance(item, str) else item.strip()
                if item_str:
                    cleaned.append(item_str)
            return cleaned

        profile['competences_techniques'] = clean_list(profile['competences_techniques'])
        profile['competences_soft'] = clean_list(profile['competences_soft'])
        profile['experiences'] = clean_list(profile['experiences'])
        profile['projets'] = clean_list(profile['projets'])

        return profile
# ============================================================
# HYBRID JOB SCRAPER AVEC EMBEDDINGS + SCORING MULTICRITÈRE
# ============================================================

class EnhancedHybridJobScraper:
    def __init__(self, apify_api_key, embedding_model=None, similarity_threshold=0.3):
        self.client = ApifyClient(apify_api_key)
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.sources_used = []
        self.last_jobs_found = []
        self.raw_jobs_found = []

    def search_jobs(self, user_profile, preferences):
        """Nouvelle architecture hybride avec embeddings + scoring multicritère"""
        location = user_profile.get('localisation', '')
        job_title = user_profile.get('job_title', '')

        if not location or not job_title:
            raise Exception("Localisation ou job title manquant")

        print(f"🔍 RECHERCHE HYBRIDE: '{job_title}' à '{location}'")

        # 1. Scraping LinkedIn
        print("\n1. 🔗 Scraping LinkedIn via Apify...")
        linkedin_jobs = self._search_apify_linkedin(job_title, location, user_profile)

        if not linkedin_jobs:
            raise Exception("Aucun job trouvé sur LinkedIn")

        print(f"📊 Total jobs LinkedIn bruts: {len(linkedin_jobs)}")

        # 2. Sélection hybride des meilleurs jobs
        print(f"\n2. 🎯 SÉLECTION HYBRIDE (Embeddings + Scoring)...")
        top_jobs = self._hybrid_job_selection(linkedin_jobs, user_profile)

        print(f"✅ Top jobs sélectionnés: {len(top_jobs)}")
        self.last_jobs_found = top_jobs.copy()

        return top_jobs

    def _hybrid_job_selection(self, jobs, user_profile):
        """Sélection hybride: 60% embeddings + 40% scoring multicritère"""
        if len(jobs) <= 3:
            return jobs[:3]  # Si peu de jobs, pas besoin de sélection complexe

        print("   🔄 Calcul des scores hybrides...")

        # Calcul du Score A: Similarité sémantique avec embeddings
        similarity_scores = self._calculate_semantic_similarity(jobs, user_profile)

        # Calcul du Score B: Scoring multicritère classique
        multicriteria_scores = self._calculate_multicriteria_scores(jobs, user_profile)

        # Combinaison: Score final = 0.6×A + 0.4×B
        hybrid_scored_jobs = []

        for job in jobs:
            job_id = job.get('url', job.get('title', ''))

            # Score A: Similarité sémantique (0-1)
            score_a = similarity_scores.get(job_id, 0.0)

            # Score B: Scoring multicritère (0-1)
            score_b = multicriteria_scores.get(job_id, {}).get('total_score', 0.0)

            # Score final hybride
            final_score = (0.2 * score_a) + (0.8 * score_b)

            hybrid_scored_jobs.append({
                'job': job,
                'final_score': round(final_score, 3),
                'score_breakdown': {
                    'similarity_score': round(score_a, 3),
                    'multicriteria_score': round(score_b, 3),
                    'tech_matches': multicriteria_scores.get(job_id, {}).get('tech_matches', []),
                    'breakdown': multicriteria_scores.get(job_id, {}).get('breakdown', {})
                }
            })

        # Trier par score final et prendre top 3
        hybrid_scored_jobs.sort(key=lambda x: x['final_score'], reverse=True)
        top_3_hybrid = hybrid_scored_jobs[:3]

        # Afficher les résultats
        self._display_hybrid_results(top_3_hybrid)

        return [item['job'] for item in top_3_hybrid]

    def _calculate_semantic_similarity(self, jobs, user_profile):
        """Calcule la similarité sémantique entre le profil et les jobs"""
        print("   🧠 Calcul similarité sémantique...")

        # Créer le texte du profil pour les embeddings
        profile_text = self._create_profile_text(user_profile)

        similarity_scores = {}

        for job in jobs:
            # Créer le texte du job
            job_text = self._create_job_text(job)

            # Calculer la similarité (méthode fallback si pas d'embedding model)
            similarity = self._calculate_text_similarity_fallback(profile_text, job_text)

            job_id = job.get('url', job.get('title', ''))
            similarity_scores[job_id] = similarity

        return similarity_scores

    def _calculate_multicriteria_scores(self, jobs, user_profile):
        """Calcule les scores multicritères classiques"""
        print("   📊 Calcul scores multicritères...")

        multicriteria_scores = {}

        for job in jobs:
            score_breakdown = self._calculate_single_job_score(job, user_profile)
            job_id = job.get('url', job.get('title', ''))
            multicriteria_scores[job_id] = score_breakdown

        return multicriteria_scores

    def _calculate_single_job_score(self, job, user_profile):
        """Calcule le score pour un seul job (version simplifiée et générique)"""
        job_text = (job.get('description', '') + ' ' + job.get('title', '')).lower()

        user_skills = [skill.lower().strip() for skill in user_profile.get('competences_techniques', [])]
        user_soft_skills = [skill.lower().strip() for skill in user_profile.get('competences_soft', [])]

        total_score = 0.0
        tech_matches = []

        # 1. Compétences techniques (50%)
        tech_score = 0.0
        for skill in user_skills:
            skill_clean = skill.strip().lower()
            if len(skill_clean) < 2:
                continue

            # Recherche flexible
            if (re.search(r'\b' + re.escape(skill_clean) + r'\b', job_text, re.IGNORECASE) or
                skill_clean in job_text):
                tech_matches.append(skill_clean)
                tech_score += 0.15

        tech_score = min(tech_score, 0.7)
        total_score += tech_score

        # 2. Titre et domaine (30%)
        title_score = 0.0
        user_title = user_profile.get('job_title', '').lower()
        job_title = job.get('title', '').lower()

        # Correspondance de mots-clés dans les titres
        user_title_words = set(re.findall(r'\b\w+\b', user_title))
        job_title_words = set(re.findall(r'\b\w+\b', job_title))
        common_words = user_title_words.intersection(job_title_words)

        if common_words:
            title_score = min(len(common_words) * 0.1, 0.1)

        total_score += title_score

        # 3. Compétences soft (20%)
        soft_score = 0.0
        for soft_skill in user_soft_skills:
            if soft_skill.lower() in job_text:
                soft_score += 0.1

        soft_score = min(soft_score, 0.2)
        total_score += soft_score

        return {
            'total_score': min(total_score, 1.0),
            'tech_matches': tech_matches,
            'breakdown': {
                'tech_score': round(tech_score, 3),
                'title_score': round(title_score, 3),
                'soft_score': round(soft_score, 3)
            }
        }

    def _calculate_text_similarity_fallback(self, text1, text2):
        """
        Calcule la similarité textuelle sans modèle d'embedding
        Méthode fallback basée sur TF-IDF simplifié
        """
        # Tokenization simple
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))

        if not words1 or not words2:
            return 0.0

        # Mots communs (Jaccard similarity)
        common_words = words1.intersection(words2)
        all_words = words1.union(words2)

        similarity = len(common_words) / len(all_words) if all_words else 0.0

        # Ajustement pour les textes courts
        return min(similarity * 1.5, 1.0)

    def _create_profile_text(self, user_profile):
        """Crée un texte représentatif du profil - CORRIGÉ"""
        components = []
        
        # Job title
        if user_profile.get('job_title'):
            components.append(str(user_profile.get('job_title', '')))
        
        # Compétences techniques
        tech_skills = user_profile.get('competences_techniques', [])
        if isinstance(tech_skills, list):
            components.append(' '.join([str(s) for s in tech_skills if s]))
        elif tech_skills:
            components.append(str(tech_skills))
        
        # Compétences soft
        soft_skills = user_profile.get('competences_soft', [])
        if isinstance(soft_skills, list):
            components.append(' '.join([str(s) for s in soft_skills if s]))
        elif soft_skills:
            components.append(str(soft_skills))
        
        # Formation
        if user_profile.get('formation'):
            components.append(str(user_profile.get('formation', '')))
        
        # Objectif professionnel
        if user_profile.get('objectif_professionnel'):
            components.append(str(user_profile.get('objectif_professionnel', '')))
        
        # Expériences
        experiences = user_profile.get('experiences', [])
        if isinstance(experiences, list):
            components.append(' '.join([str(e) for e in experiences if e]))
        elif experiences:
            components.append(str(experiences))
        
        # Filtrer les composants vides et joindre
        clean_components = [c for c in components if c and c.strip()]
        return ' '.join(clean_components).lower()

    def _create_job_text(self, job):
        """Crée un texte représentatif du job - CORRIGÉ"""
        components = []
        
        # Title
        if job.get('title'):
            components.append(str(job.get('title', '')))
        
        # Description
        if job.get('description'):
            desc = job.get('description', '')
            # Si la description est une liste, la joindre
            if isinstance(desc, list):
                components.append(' '.join([str(d) for d in desc if d]))
            else:
                components.append(str(desc))
        
        # Company
        if job.get('company'):
            components.append(str(job.get('company', '')))
        
        # Filtrer les composants vides et joindre
        clean_components = [c for c in components if c and c.strip()]
        return ' '.join(clean_components).lower()

    def _display_hybrid_results(self, hybrid_results):
        """Affiche les résultats de la sélection hybride"""
        print(f"\n   🏆 TOP 3 HYBRIDE SELECTIONNÉS:")
        for i, result in enumerate(hybrid_results):
            job = result['job']
            breakdown = result['score_breakdown']

            print(f"   {i+1}. Score: {result['final_score']:.3f} "
                  f"(sémantique:{breakdown['similarity_score']:.3f} "
                  f"multicritère:{breakdown['multicriteria_score']:.3f})")
            print(f"      💼 {job['title'][:60]}...")
            if breakdown['tech_matches']:
                print(f"      🔧 Tech: {', '.join(breakdown['tech_matches'][:3])}")

    # Garder les méthodes existantes (_search_apify_linkedin, _parse_linkedin_item, etc.)
    def _search_apify_linkedin(self, job_title, location, user_profile):
        """Scraping LinkedIn via Apify - 10 jobs maximum"""
        try:
            run_input = {
                "query": job_title,
                "location": location,
                "jobsToFetch": 2 ,  # Augmenté pour avoir plus de choix
                "proxyConfiguration": {"useApifyProxy": True}
            }

            run = self.client.actor("PeTP8M7vkdTthJvqk").call(run_input=run_input)
            all_jobs = []

            for item in list(self.client.dataset(run["defaultDatasetId"]).iterate_items()):
                job = self._parse_linkedin_item(item)
                if job:
                    all_jobs.append(job)

            return all_jobs[:10]

        except Exception as e:
            print(f"❌ Erreur Apify LinkedIn: {e}")
            return []

    def _parse_linkedin_item(self, item):
        """Parse item LinkedIn Apify"""
        try:
            title = item.get("title", "") or item.get("jobTitle", "")
            if not title:
                return None

            company_data = item.get("hiringOrganization", {}) or item.get("company", "")
            company = company_data.get("name", "") if isinstance(company_data, dict) else str(company_data)

            description_data = item.get("jobDescription", "") or item.get("description", "")
            description = " ".join(description_data) if isinstance(description_data, list) else str(description_data)

            job_link = item.get("link") or item.get("jobUrl") or item.get("applyLink") or ""

            return {
                "title": title,
                "company": company,
                "location": str(item.get("jobLocation", "")),
                "description": description[:4000],
                "url": job_link,
                "job_link": job_link,
                "source": "LinkedIn"
            }
        except:
            return None

# ============================================================
# JOB ANALYZER AVEC GROQ - VERSION ADAPTÉE
# ============================================================

class JobAnalyzer:
    def __init__(self, groq_processor):
        self.groq = groq_processor  # 🔥 Remplacement par Groq

    def analyze_job(self, job_data):
        """Analyse une offre d'emploi avec Groq et AFFICHE le JSON extrait"""
        job_text = f"{job_data['title']} {job_data['description']}"[:4000]

        prompt = f"""
        EXTRAIT TOUTES les informations suivantes de cette offre d'emploi de manière EXHAUSTIVE :

        OFFRE D'EMPLOI:
        {job_text}

        FORMAT JSON STRICT:
{{
    "competences_techniques": ["liste COMPLÈTE de toutes les compétences techniques mentionnées"],
    "competences_soft": ["liste COMPLÈTE de toutes les compétences comportementales"],
    "experiences": ["liste COMPLÈTE des types d'expérience recherchés"],
    "formation": "diplôme requis",
    "objectif_professionnel": "type de profil recherché",
    "localisation": "lieu de travail"
}}

        RÈGLES IMPORTANTES:
        - Extrait TOUTES les compétences techniques (même implicites)
        - Inclus les technologies, méthodologies, cas d'usage
        - Extrait SEULEMENT les informations présentes dans l'offre
        - Capture TOUTES les expériences demandées
        - Sois EXHAUSTIF - ne laisse rien d'important
        - Pour les compétences: prends les mots-clés techniques mentionnés
        - Inclus les valeurs d'entreprise comme compétences soft
        - Retourne UNIQUEMENT du JSON
        """

        try:
            # 🔥 UTILISATION DE GROQ AU LIEU DE LLM_PROCESSOR
            response = self.groq.generate_text(prompt, max_tokens=400)

            # Nettoyage
            response = response.strip()
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0].strip()
            elif '```' in response:
                parts = response.split('```')
                response = parts[1].strip() if len(parts) > 1 else parts[0].strip()

            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                response = response[start_idx:end_idx]

            job_profile = json.loads(response)

            # Validation
            job_profile = self._validate_job_profile(job_profile)
            job_profile['raw_job_text'] = job_text

            # ✅ AFFICHAGE DU JSON JOB EXTRAIT
            print("🎯 JSON JOB EXTRAIT (Groq 70B):")
            print(json.dumps(job_profile, indent=2, ensure_ascii=False))
            print("="*50)

            return job_profile

        except Exception as e:
            print(f"❌ Erreur analyse job avec Groq: {e}")
            return None

    def _validate_job_profile(self, profile):
        """Valide et nettoie le profil job"""
        default_profile = {
            "competences_techniques": [],
            "competences_soft": [],
            "experiences": [],
            "formation": "",
            "objectif_professionnel": "",
            "localisation": ""
        }

        for key, default in default_profile.items():
            if key not in profile:
                profile[key] = default
            elif profile[key] is None:
                profile[key] = default

        profile['competences_techniques'] = [comp.strip() for comp in profile['competences_techniques'] if comp and comp.strip()]
        profile['competences_soft'] = [comp.strip() for comp in profile['competences_soft'] if comp and comp.strip()]

        return profile

# ============================================================
# INTELLIGENT JOB MATCHER - SCORING CONTEXTUEL PRÉCIS
# ============================================================

class IntelligentJobMatcher:
    def __init__(self, groq_processor):
        self.job_analyzer = JobAnalyzer(groq_processor)
        self.groq = groq_processor

    def find_best_match(self, user_profile, jobs):
        """Trouve le meilleur match avec scoring contextuel précis"""
        if not jobs:
            print("❌ Aucun job à analyser")
            return None

        print(f"🔍 DÉBUT ANALYSE CONTEXTUELLE PRÉCISE sur {len(jobs)} jobs")

        # Étape 1: Analyse contextuelle détaillée
        print("\n🎯 ÉTAPE 1: Analyse contextuelle détaillée...")
        analyzed_jobs = self._detailed_contextual_analysis(user_profile, jobs)

        if not analyzed_jobs:
            print("❌ Aucune analyse contextuelle valide")
            return None

        # Étape 2: Affichage du classement détaillé
        self._display_detailed_ranking(analyzed_jobs)

        # Étape 3: Retour du meilleur match
        best_match = analyzed_jobs[0]
        self._display_best_match_details(best_match)

        return best_match

    def _detailed_contextual_analysis(self, user_profile, jobs):
        """Analyse contextuelle détaillée avec scoring précis"""
        analyzed_jobs = []

        for i, job in enumerate(jobs[:5]):  # Limiter à 5 pour la qualité
            print(f"   📊 Analyse {i+1}/{min(5, len(jobs))}: {job['title'][:45]}...")

            try:
                # Analyse du job avec Groq
                job_analysis = self.job_analyzer.analyze_job(job)
                if not job_analysis:
                    print(f"      ⚠️ Analyse du job échouée")
                    continue

                # Calcul du score contextuel détaillé
                contextual_score, score_details, reasoning = self._calculate_precise_contextual_score(
                    user_profile, job_analysis, job
                )

                analyzed_jobs.append({
                    "job": job,
                    "job_analysis": job_analysis,
                    "contextual_score": contextual_score,
                    "score_details": score_details,
                    "reasoning": reasoning,
                    "final_score": contextual_score
                })

                # Affichage intermédiaire détaillé
                self._display_intermediate_score(job, contextual_score, score_details)

            except Exception as e:
                print(f"      ❌ Erreur analyse: {e}")
                continue

        # Trier par score contextuel
        analyzed_jobs.sort(key=lambda x: x["contextual_score"], reverse=True)
        return analyzed_jobs

    def _calculate_precise_contextual_score(self, user_profile, job_analysis, job):
        """Calcule un score contextuel précis avec composantes détaillées"""

        # Prompt détaillé pour une analyse précise
        prompt = self._create_precise_scoring_prompt(user_profile, job_analysis, job)

        try:
            response = self.groq.generate_text(prompt, max_tokens=1000)
            return self._parse_precise_score_response(response, user_profile, job_analysis)

        except Exception as e:
            print(f"      ❌ Erreur Groq: {e}")
            return self._calculate_fallback_score(user_profile, job_analysis, job)

    def _create_precise_scoring_prompt(self, user_profile, job_analysis, job):
        """Crée un prompt détaillé pour le scoring précis"""
        return f"""
        Tu es un expert en recrutement technique. Analyse la compatibilité entre le candidat et le poste avec une évaluation PRÉCISE et DÉTAILLÉE.

        ========== PROFIL DU CANDIDAT ==========
        📍 Titre actuel: {user_profile.get('job_title', 'Non spécifié')}
        🔧 Compétences techniques: {', '.join(user_profile.get('competences_techniques', []))}
        🤝 Compétences soft: {', '.join(user_profile.get('competences_soft', []))}
        📈 Expériences: {len(user_profile.get('experiences', []))} postes
        🎓 Formation: {user_profile.get('formation', 'Non spécifié')}
        🎯 Objectif: {user_profile.get('objectif_professionnel', 'Non spécifié')}

        ========== OFFRE D'EMPLOI ==========
        📍 Poste: {job['title']}
        🏢 Entreprise: {job.get('company', 'Non spécifié')}
        🔧 Compétences requises: {', '.join(job_analysis.get('competences_techniques', [])[:15])}
        🤝 Soft skills recherchés: {', '.join(job_analysis.get('competences_soft', [])[:10])}
        📋 Description: {job.get('description', '')[:1500]}

        ========== CRITÈRES DE SCORING DÉTAILLÉS ==========

        1. 🔧 ADÉQUATION TECHNIQUE (40 points maximum)
        • Correspondance exacte des compétences techniques
        • Pertinence des technologies maîtrisées
        • Niveau d'expertise requis vs. actuel

        2. 📈 ADÉQUATION EXPÉRIENCE (25 points maximum)
        • Nombre d'années d'expérience pertinente
        • Complexité et envergure des projets réalisés
        • Similarité des responsabilités passées

        3. 🎯 ADÉQUATION FORMATION & COMPÉTENCES (15 points maximum)
        • Niveau de formation requis vs. actuel
        • Certifications et spécialisations
        • Compétences transversales

        4. 🔄 POTENTIEL D'ADAPTATION (10 points maximum)
        • Diversité des expériences
        • Capacité d'apprentissage démontrée
        • Flexibilité et polyvalence

        5. 🤝 ADÉQUATION CULTURELLE & SOFT SKILLS (10 points maximum)
        • Alignement des valeurs et soft skills
        • Compatibilité avec la culture d'entreprise
        • Capacité à travailler en équipe

        ========== CONSIGNES DE SCORING ==========
        • Sois RÉALISTE et EXIGEANT dans ton évaluation
        • Un score de 90-100/100 = correspondance EXCEPTIONNELLE
        • Un score de 75-89/100 = correspondance BONNE à TRÈS BONNE
        • Un score de 60-74/100 = correspondance MOYENNE à BONNE
        • Un score de 40-59/100 = correspondance LIMITÉE
        • Un score < 40/100 = correspondance FAIBLE

        ========== FORMAT DE RÉPONSE OBLIGATOIRE ==========
        SCORE_TOTAL: [score/100]
        COMPOSANTE_TECHNIQUE: [score/40] - [justification technique]
        COMPOSANTE_EXPERIENCE: [score/25] - [justification expérience]
        COMPOSANTE_FORMATION: [score/15] - [justification formation]
        COMPOSANTE_ADAPTATION: [score/10] - [justification adaptation]
        COMPOSANTE_CULTURELLE: [score/10] - [justification culturelle]
        ANALYSE_SYNTHESE: [analyse globale en 3-4 phrases]
        POINTS_FORTS: [2-3 points forts principaux]
        POINTS_FAIBLES: [2-3 points d'amélioration]
        RECOMMANDATION: [recommandation réaliste]
        """

    def _parse_precise_score_response(self, response, user_profile, job_analysis):
        """Parse la réponse détaillée et calcule le score final"""
        try:
            # Extraction des scores par composante
            components = {
                'technique': self._extract_component_score(response, 'COMPOSANTE_TECHNIQUE', 40),
                'experience': self._extract_component_score(response, 'COMPOSANTE_EXPERIENCE', 25),
                'formation': self._extract_component_score(response, 'COMPOSANTE_FORMATION', 15),
                'adaptation': self._extract_component_score(response, 'COMPOSANTE_ADAPTATION', 10),
                'culturelle': self._extract_component_score(response, 'COMPOSANTE_CULTURELLE', 10)
            }

            # Score total
            total_score_match = re.search(r'SCORE_TOTAL:\s*(\d+(?:\.\d+)?)\s*\/\s*100', response, re.IGNORECASE)
            total_score = float(total_score_match.group(1)) / 100.0 if total_score_match else None

            # Validation et calcul du score final
            final_score, validated_components = self._validate_and_calculate_final_score(components, total_score, response)

            # Extraction des justifications
            reasoning = self._extract_detailed_reasoning(response)
            score_details = self._create_score_details(validated_components, reasoning)

            return final_score, score_details, reasoning

        except Exception as e:
            print(f"      ⚠️ Erreur parsing réponse: {e}")
            return self._calculate_fallback_score(user_profile, job_analysis, job)

    def _extract_component_score(self, response, component_name, max_score):
        """Extrait le score d'une composante spécifique"""
        pattern = f"{component_name}:\\s*(\\d+(?:\\.\\d+)?)\\s*\\/\\s*{max_score}"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None

    def _validate_and_calculate_final_score(self, components, total_score, response):
        """Valide et calcule le score final de manière cohérente"""
        # Calculer le score à partir des composantes
        calculated_score = 0.0
        total_max = 0
        validated_components = {}

        for comp_name, comp_score in components.items():
            if comp_score is not None:
                validated_components[comp_name] = comp_score
                # Conversion en pourcentage du total (100 points)
                if comp_name == 'technique':
                    calculated_score += (comp_score / 40) * 40
                elif comp_name == 'experience':
                    calculated_score += (comp_score / 25) * 25
                elif comp_name == 'formation':
                    calculated_score += (comp_score / 15) * 15
                elif comp_name == 'adaptation':
                    calculated_score += (comp_score / 10) * 10
                elif comp_name == 'culturelle':
                    calculated_score += (comp_score / 10) * 10

        calculated_total = calculated_score

        # Validation de cohérence
        if total_score is not None:
            # Vérifier la cohérence entre le score total et les composantes
            score_diff = abs((total_score * 100) - calculated_total)
            if score_diff > 15:  # Écart trop important
                print(f"      ⚠️ Incohérence score: total={total_score*100:.1f}, calculé={calculated_total:.1f}")
                # Utiliser une moyenne pondérée
                final_total = (calculated_total * 0.7) + (total_score * 100 * 0.3)
            else:
                final_total = (calculated_total * 0.4) + (total_score * 100 * 0.6)
        else:
            final_total = calculated_total

        # Ajustements basés sur le contenu textuel
        final_total = self._apply_textual_adjustments(final_total, response)

        # Conversion en score 0-1
        final_score_normalized = final_total / 100.0

        return final_score_normalized, validated_components

    def _apply_textual_adjustments(self, score, response):
        """Ajuste le score basé sur l'analyse textuelle"""
        response_lower = response.lower()

        # Indicateurs négatifs (réduction de score)
        negative_indicators = {
            'mais': 0.95, 'cependant': 0.93, 'néanmoins': 0.92, 'toutefois': 0.94,
            'manque': 0.85, 'limité': 0.80, 'faible': 0.75, 'insuffisant': 0.70,
            'écart': 0.88, 'défaut': 0.85, 'problème': 0.90, 'difficulté': 0.92
        }

        # Indicateurs positifs (augmentation de score)
        positive_indicators = {
            'excellent': 1.10, 'parfait': 1.15, 'idéal': 1.12, 'exceptionnel': 1.18,
            'forte adéquation': 1.08, 'haute compatibilité': 1.10, 'correspondance parfaite': 1.20,
            'très bon': 1.05, 'solide': 1.04, 'pertinent': 1.03
        }

        # Application des ajustements
        adjusted_score = score

        for indicator, factor in negative_indicators.items():
            if indicator in response_lower:
                adjusted_score *= factor
                print(f"      📉 Ajustement négatif: {indicator} (x{factor})")

        for indicator, factor in positive_indicators.items():
            if indicator in response_lower:
                adjusted_score *= factor
                print(f"      📈 Ajustement positif: {indicator} (x{factor})")

        return min(adjusted_score, 95)  # Limiter à 95%

    def _calculate_fallback_score(self, user_profile, job_analysis, job):
        """Calcule un score de fallback basé sur l'analyse automatique"""
        print("      🔄 Utilisation du score de fallback")

        # Analyse basique des correspondances
        user_skills = set(skill.lower() for skill in user_profile.get('competences_techniques', []))
        job_skills = set(skill.lower() for skill in job_analysis.get('competences_techniques', []))

        # Correspondances exactes
        exact_matches = len(user_skills.intersection(job_skills))

        # Score basé sur les correspondances
        if job_skills:
            match_ratio = exact_matches / len(job_skills)
        else:
            match_ratio = 0

        # Ajustement basé sur l'expérience
        exp_adjustment = min(len(user_profile.get('experiences', [])) * 0.05, 0.2)

        base_score = (match_ratio * 0.7) + exp_adjustment
        final_score = max(0.1, min(0.8, base_score))  # Bornes réalistes

        reasoning = "Score calculé automatiquement basé sur la correspondance des compétences techniques"
        score_details = {
            'technique': match_ratio * 40,
            'experience': exp_adjustment * 25,
            'formation': 0,
            'adaptation': 0,
            'culturelle': 0
        }

        return final_score, score_details, reasoning

    def _extract_detailed_reasoning(self, response):
        """Extrait le raisonnement détaillé de la réponse"""
        try:
            # Chercher l'analyse synthèse
            synthèse_match = re.search(
                r'ANALYSE_SYNTHESE:\s*(.+?)(?=POINTS_FORTS:|RECOMMANDATION:|$)',
                response,
                re.IGNORECASE | re.DOTALL
            )

            if synthèse_match:
                reasoning = synthèse_match.group(1).strip()
                # Nettoyer
                reasoning = re.sub(r'\s+', ' ', reasoning)
                return reasoning[:400] + "..." if len(reasoning) > 400 else reasoning

            # Fallback
            lines = [line.strip() for line in response.split('\n') if len(line.strip()) > 50]
            if lines:
                return lines[0][:300] + "..."

            return "Analyse contextuelle réalisée"

        except Exception:
            return "Évaluation basée sur l'analyse des correspondances"

    def _create_score_details(self, components, reasoning):
        """Crée le détail des scores pour l'affichage"""
        return {
            'components': components,
            'reasoning': reasoning,
            'total_breakdown': f"Technique: {components.get('technique', 0):.1f}/40, " +
                              f"Expérience: {components.get('experience', 0):.1f}/25, " +
                              f"Formation: {components.get('formation', 0):.1f}/15"
        }

    def _display_intermediate_score(self, job, score, score_details):
        """Affiche le score intermédiaire avec détails"""
        print(f"      ✅ Score: {score:.3f}")
        if 'components' in score_details:
            comp = score_details['components']
            print(f"         🔧 Technique: {comp.get('technique', 0):.1f}/40")
            print(f"         📈 Expérience: {comp.get('experience', 0):.1f}/25")

    def _display_detailed_ranking(self, analyzed_jobs):
        """Affiche le classement détaillé"""
        print(f"\n🏆 CLASSEMENT FINAL DÉTAILLÉ:")

        for i, match in enumerate(analyzed_jobs):
            score = match['contextual_score']
            job = match['job']
            details = match['score_details']

            print(f"   {i+1}. 🎯 Score: {score:.3f} ({score*100:.1f}%)")
            print(f"      💼 {job['title'][:60]}...")
            print(f"      🏢 {job.get('company', 'Non spécifié')}")

            if 'total_breakdown' in details:
                print(f"      📊 {details['total_breakdown']}")

            print(f"      💡 {match['reasoning'][:100]}...")
            print()

    def _display_best_match_details(self, best_match):
        """Affiche les détails du meilleur match"""
        print(f"🎯 MEILLEUR MATCH FINAL:")
        print(f"   📍 Poste: {best_match['job']['title']}")
        print(f"   🏢 Entreprise: {best_match['job'].get('company', 'Non spécifié')}")
        print(f"   🎯 Score: {best_match['contextual_score']:.3f} ({best_match['contextual_score']*100:.1f}%)")
        print(f"   📈 Analyse: {best_match['reasoning']}")

        # Détails des composantes de score
        if 'components' in best_match['score_details']:
            comp = best_match['score_details']['components']
            print(f"   📊 DÉTAIL DU SCORE:")
            print(f"      🔧 Technique: {comp.get('technique', 0):.1f}/40")
            print(f"      📈 Expérience: {comp.get('experience', 0):.1f}/25")
            print(f"      🎓 Formation: {comp.get('formation', 0):.1f}/15")
            print(f"      🔄 Adaptation: {comp.get('adaptation', 0):.1f}/10")
            print(f"      🤝 Culturel: {comp.get('culturelle', 0):.1f}/10")



# ============================================================
# CELLULE 9 - Système d'Embedding (léger)
# ============================================================

class EmbeddingProcessor:
    def __init__(self):
        print("🔄 Chargement modèle embedding léger...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding chargé")

    def get_embedding(self, text):
        """Génère embedding"""
        if isinstance(text, list):
            text = ' '.join(text)
        return self.model.encode([text])[0]

    def calculate_similarity(self, text1, text2):
        """Calcule similarité cosinus"""
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return float(similarity)  # Conversion pour JSON


# ============================================================
# CV ENHANCER AVEC RAG - VERSION GROQ AVEC REFORMULATION CONTEXTUELLE
# ============================================================

class EthicalRAGCVEnhancer:
    def __init__(self, groq_processor):
        self.groq = groq_processor  # Remplace llm_processor par groq_processor
        self.vector_store = None
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        print("✅ Ethical RAG CV Enhancer avec Groq initialisé")

    def enhance_cv_ethically(self, user_profile, job_profile):
     """
     Améliore le CV avec RAG - AFFICHE TOUJOURS LA RÉPONSE DU LLM
     """
     print("🔍 RAG: Analyse de la description du job...")

     try:
        # Étape 1: Préparer contexte job
         job_context = self._prepare_job_context(job_profile)

        # Étape 2: Initialiser RAG
         self._initialize_rag_with_job(job_context)

        # Étape 3: Recherche RAG
         rag_advice = self._get_rag_advice(user_profile, job_profile)

        # Étape 4: Générer enhancement avec Groq
         enhanced_cv = self._generate_enhanced_cv(user_profile, job_profile, rag_advice)

        # Étape 5: Validation éthique
         if enhanced_cv and isinstance(enhanced_cv, dict) and 'competences_techniques' in enhanced_cv:
             if not self._validate_ethical_enhancement(user_profile, enhanced_cv):
                 print("🚨 COMPÉTENCES INVENTÉES DÉTECTÉES (résultat conservé)")
             else:
                 print("✅ Reformulation éthique validée")
        
        # CORRECTION: S'assurer que enhanced_cv contient tous les champs nécessaires
         if enhanced_cv and isinstance(enhanced_cv, dict):
            # Fusionner avec le profil original pour éviter les champs manquants
             complete_cv = {**user_profile, **enhanced_cv}
             return complete_cv
        
         return enhanced_cv if enhanced_cv else user_profile

     except Exception as e:
         print(f"❌ Erreur RAG critique: {e}")
        # Retourner CV original en cas d'erreur critique
         return user_profile



    def _prepare_job_context(self, job_profile):
        """Prépare le contexte du job"""
        job_text = f"""
        JOB TITLE: {job_profile.get('objectif_professionnel', '')}
        REQUIRED TECHNICAL SKILLS: {', '.join(job_profile.get('competences_techniques', []))}
        REQUIRED SOFT SKILLS: {', '.join(job_profile.get('competences_soft', []))}
        REQUIRED EXPERIENCE: {', '.join(job_profile.get('experiences', []))}
        REQUIRED EDUCATION: {job_profile.get('formation', '')}
        JOB DESCRIPTION: {job_profile.get('raw_job_text', '')[:4000]}
        """
        return job_text

    def _initialize_rag_with_job(self, job_context):
        """Initialise RAG avec contexte job"""
        job_docs = [Document(page_content=job_context, metadata={"source": "job_description"})]

        enhancement_techniques = [
            Document(
                page_content="""
                CONTEXTUAL ENHANCEMENT TECHNIQUES:
                - VOCABULARY ALIGNMENT: Use the same terminology as the job description
                - SKILL MAPPING: Map existing skills to job requirements using similar wording
                - EXPERIENCE REFRAMING: Rephrase experiences to highlight relevant aspects for the specific job
                - HIDDEN SKILL EXTRACTION: Infer implied technical skills from experiences that match job requirements
                - KEYWORD INTEGRATION: Naturally incorporate keywords from the job description
                - PROFESSIONAL PHRASING: Use industry-standard professional terminology
                - CLARITY ENHANCEMENT: Replace vague descriptions with precise, measurable, and domain-specific terms
                - DOMAIN GENERALIZATION: Apply these enhancements to *any* professional field (data science, marketing, finance, AI, etc.), not just a specific one.

                EXAMPLES:
                 1. Job requires: "Generative AI experience"
           CV contains: "Worked with LLMs, Stable Diffusion, or NLP models"
           → Rephrase as: "Experience with Generative AI including LLMs and text-to-image models (Stable Diffusion)"

        2. Job requires: "Power BI dashboards and KPIs"
           CV contains: "Created Excel dashboards for sales tracking"
           → Rephrase as: "Developed interactive Power BI dashboards to monitor sales KPIs"

        3. Job requires: "ETL pipelines and SQL queries"
           CV contains: "Wrote Python scripts to clean and move data"
           → Rephrase as: "Implemented ETL pipelines and SQL queries for data extraction and transformation using Python"

        4. Job requires: "Computer Vision / Image Processing"
           CV contains: "Worked with OpenCV to detect objects"
           → Rephrase as: "Applied computer vision techniques using OpenCV for object detection tasks"

        5. Job requires: "Marketing automation and content generation"
           CV contains: "Created scripts to generate social media posts"
           → Rephrase as: "Developed automated content generation workflows for marketing campaigns"

        6. Job requires: "Financial forecasting and analysis"
           CV contains: "Analyzed historical sales data and trends"
           → Rephrase as: "Performed financial forecasting and trend analysis to support business decisions"

        7. Job requires: "Healthcare data analysis and dashboards"
           CV contains: "Processed patient data for reporting"
           → Rephrase as: "Analyzed healthcare datasets and developed dashboards for clinical reporting and KPIs"

        8. Job requires: "AI/ML pipeline deployment"
           CV contains: "Deployed machine learning models using Flask"
           → Rephrase as: "Deployed AI/ML models in production pipelines using Flask and containerized services"

        9. Job requires: "Prompt engineering and LLM fine-tuning"
           CV contains: "Fine-tuned transformer models for text classification"
           → Rephrase as: "Applied prompt engineering and fine-tuning techniques on transformer-based LLMs for domain-specific text tasks"

        10. Job requires: "Data storytelling for business insights"
            CV contains: "Presented analysis results to management"
            → Rephrase as: "Translated analytical results into actionable business insights through clear data storytelling"
                """,
                metadata={"source": "contextual_enhancement", "type": "techniques"}
            ),
            Document(
                page_content="""
                ETHICAL REFORMULATION PRINCIPLES:
                1. CONTEXTUAL ADAPTATION: Adapt existing skills to job context without inventing
                2. TERMINOLOGY MATCHING: Use job description vocabulary for existing skills
                3. HIDDEN SKILL EXTRACTION: Infer implied technical skills from experiences that match job requirements
                4. RELEVANCE HIGHLIGHTING: Emphasize aspects most relevant to the position
                5. PROFESSIONAL WORDING: Upgrade to professional business language
                6. QUANTIFICATION: Add metrics only if implied in original experience
                7. DO not invent skills non existant in orignal CV
                """,
                metadata={"source": "ethical_principles", "type": "principles"}
            )
        ]

        all_docs = job_docs + enhancement_techniques

        self.vector_store = Chroma.from_documents(
            documents=all_docs,
            embedding=self.embeddings,
            persist_directory="./rag_job_context"
        )

    def _get_rag_advice(self, user_profile, job_profile):
        """Récupère conseils RAG"""
        rag_query = f"""
        How to contextually adapt this CV to match the job description vocabulary? and How to intelligently extract hidden skills from candidate experiences that match job requirements?

        CANDIDATE PROFILE:
        - Current skills: {', '.join(user_profile.get('competences_techniques', []))}
        - Current experiences: {' | '.join(user_profile.get('experiences', [])[:2])}

        JOB REQUIREMENTS:
        - Required skills: {', '.join(job_profile.get('competences_techniques', []))}
        - Job title: {job_profile.get('objectif_professionnel', '')}
        - Key job description terms: {self._extract_key_terms(job_profile.get('raw_job_text', ''))}

        Goal: Reformulate candidate's existing skills using job description vocabulary and identify implied skills in experiences and map them to job requirements.
        """

        relevant_docs = self.vector_store.similarity_search(rag_query, k=4)

        advice = []
        for doc in relevant_docs:
            if "contextual_enhancement" in doc.metadata.get("source", ""):
                advice.append("CONTEXTUAL TECHNIQUES: " + doc.page_content[:300])
            elif "ethical_principles" in doc.metadata.get("source", ""):
                advice.append("ETHICAL PRINCIPLES: " + doc.page_content[:200])
            elif "job_description" in doc.metadata.get("source", ""):
                advice.append("JOB CONTEXT: " + self._extract_job_keywords(doc.page_content))

        return "\n".join(advice)

    def _extract_key_terms(self, job_text):
        """Extract key terms from job description"""
        words = job_text.lower().split()
        important_terms = [word for word in words if len(word) > 5 and word.isalpha()]
        return ', '.join(list(set(important_terms))[:10])

    def _extract_job_keywords(self, job_content):
        """Extract important keywords from job content"""
        lines = job_content.split('\n')
        keywords = []
        for line in lines:
            if any(keyword in line.lower() for keyword in ['skill', 'require', 'must have', 'qualification', 'experience']):
                keywords.append(line.strip())
        return ' | '.join(keywords[:3])

    def _detect_cv_language(self, cv_data):
     """Détecte la langue du CV"""
     sample_text = ""
     if cv_data.get('objectif_professionnel'):
         sample_text += cv_data.get('objectif_professionnel', '')
     if cv_data.get('experiences'):
        # Prendre les 2 premières expériences
         exps = cv_data.get('experiences', [])[:2]
         for exp in exps:
             if isinstance(exp, dict):
                 sample_text += ' ' + exp.get('description', '')
             else:
                 sample_text += ' ' + str(exp)
    
     french_indicators = ['profil','Résumé','compétences', 'expérience', 'formation', 'projets']
     english_indicators = ['profile','summary','skills', 'work experience', 'education', 'projects']
    
     sample_lower = sample_text.lower()
     french_count = sum(1 for word in french_indicators if word in sample_lower)
     english_count = sum(1 for word in english_indicators if word in sample_lower)
    
     return "french" if french_count > english_count else "english"

    def _generate_enhanced_cv(self, user_profile, job_profile, rag_advice):
        """Génère CV amélioré - AFFICHE TOUJOURS LA RÉPONSE"""

        cv_language = self._detect_cv_language(user_profile)

        # Construire le prompt contextuel
        if cv_language == "french":
            prompt = self._build_contextual_french_prompt(user_profile, job_profile, rag_advice)
        else:
            prompt = self._build_contextual_english_prompt(user_profile, job_profile, rag_advice)

        print("\n" + "="*80)
        print("📤 PROMPT ENVOYÉ AU LLM:")
        print("="*80)
        print(prompt[:2500] + "...\n")

        try:
            # Générer avec Groq (seule modification majeure)
            response = self.groq.generate_text(prompt, max_tokens=1024)

            print("="*80)
            print("📥 RÉPONSE BRUTE DU LLM (Groq 70B):")
            print("="*80)
            print(response)
            print("="*80 + "\n")

            # Essayer de parser le JSON
            enhanced_cv = self._parse_json_response(response)

            if enhanced_cv:
                print("✅ JSON parsé avec succès")
                if 'raw_cv_text' in enhanced_cv:
                  del enhanced_cv['raw_cv_text']
                if 'raw_cv_text' in user_profile:
                  del user_profile['raw_cv_text']

                final_enhanced_cv = {
                **user_profile,  # Tous les champs originaux
                **enhanced_cv   # Écraser avec les champs améliorés
                }
                return final_enhanced_cv
            else:
                print("⚠️ Parsing JSON échoué - Retour de la réponse brute")
                # Retourner structure avec réponse brute du LLM
                return {
                    "competences_techniques": [f"[RÉPONSE LLM BRUTE] {response[:200]}"],
                    "competences_soft": user_profile.get('competences_soft', []),
                    "experiences": [f"[PARSING ÉCHOUÉ] Voir _llm_raw_response pour la réponse complète"],
                    "formation": user_profile.get('formation', ''),
                    "objectif_professionnel": user_profile.get('objectif_professionnel', ''),
                    "localisation": user_profile.get('localisation', ''),
                    "nom_complet": user_profile.get('nom_complet', ''),
                    "telephone": user_profile.get('telephone', ''),
                    "email": user_profile.get('email', ''),
                    "job_title": user_profile.get('job_title', ''),
                    "_llm_raw_response": response,
                    "_parsing_error": "Le LLM n'a pas retourné un JSON valide"
                }

        except Exception as e:
            print(f"❌ Erreur génération Groq: {e}")
            return user_profile

    def _build_contextual_english_prompt(self, user_profile, job_profile, rag_advice):
        """Prompt anglais avec reformulation contextuelle"""
        # 🔥 AJOUT: Vérifier si le CV original contient des projets
        has_projects = bool(user_profile.get('projets') and len(user_profile.get('projets', [])) > 0)
        projects_warning = ""
    
        if has_projects:
         projects_warning = """
⚠️ CRITICAL WARNING: This CV contains existing PROJECTS.
Projects are MANDATORY in the output JSON.
NEVER omit or delete projects from the original CV.
"""
        return f"""
# CONTEXTUAL ENHANCEMENT GUIDANCE:
{rag_advice}
{projects_warning}

# TASK: Contextual CV Reformulation
Reformulate the candidate's CV using the job description vocabulary and context.

# ORIGINAL DATA:
- competences_techniques: {json.dumps(user_profile.get('competences_techniques', []))}
- competences_soft: {json.dumps(user_profile.get('competences_soft', []))}
- experiences: {json.dumps(user_profile.get('experiences', [])[:3])}
- projets: {json.dumps(user_profile.get('projets', []))}
- formation: "{user_profile.get('formation', '')}"
- objectif_professionnel: "{user_profile.get('objectif_professionnel', '')}"

# JOB CONTEXT:
- Position: {job_profile.get('objectif_professionnel', '')}
- Key Skills: {', '.join(job_profile.get('competences_techniques', [])[:8])}
- Job Keywords: {self._extract_key_terms(job_profile.get('raw_job_text', ''))}

# STRATEGY:
1. UNDERSTAND HIDDEN SKILLS: Analyze experiences to extract implied technical skills that match job requirements
2. SKILL INFERENCE: If experience implies a skill required by job, add it to competences_techniques
3. VOCABULARY ALIGNMENT: Use job description terminology for reformulation
4. EXPERIENCE REPHRASING: Rephrase experiences using job-specific vocabulary and context
5. SMART MAPPING: Map candidate's actual work to job requirements intelligently
6. PRESERVE TRUTH: Only add skills that are genuinely implied by experiences
7. DO NOT INVENT: Never add completely fictional skills

# HIDDEN SKILL EXTRACTION EXAMPLES:
- Experience: "Developed LLM models for marketing content generation"
  Job requires: "Generative AI experience"
  → Add: "Generative AI" to competences_techniques ✅

- Experience: "Analyzed sales data to predict trends"
  Job requires: "Predictive sales modeling"
  → Add: "Predictive Modeling" to competences_techniques ✅

- Experience: "Worked with Python for data processing"
  Job requires: "Pandas data analysis"
  → Add: "Pandas" to competences_techniques ✅

- Experience: "Created dashboards for team reporting"
  Job requires: "Data visualization"
  → Add: "Data Visualization" to competences_techniques ✅

# EXPERIENCE REPHRASING EXAMPLES:
- Original: "Application to anticipate sales"
  Job term: "Predictive sales models"
  → Rephrase: "Built predictive sales models" ✅

- Original: "Worked with customer data"
  Job term: "Customer analytics"
  → Rephrase: "Performed customer analytics" ✅

- Original: "Made reports for management"
  Job term: "Executive dashboards"
  → Rephrase: "Developed executive dashboards" ✅

# IMPORTANT: Apply these techniques to ANY domain (tech, marketing, finance, healthcare, etc.)


# RESPONSE FORMAT (exact same fields):
{{
    "competences_techniques": ["reformulated skills"],
    "competences_soft": ["reformulated soft skills"],
    "experiences": ["reformulated experiences"],
    "projets": ["reformulated projects"],
    "formation": "reformulated education",
    "objectif_professionnel": "reformulated objective",
    "localisation": "{user_profile.get('localisation', '')}",
    "nom_complet": "{user_profile.get('nom_complet', '')}",
    "telephone": "{user_profile.get('telephone', '')}",
    "email": "{user_profile.get('email', '')}",
    "job_title": "{user_profile.get('job_title', '')}"
}}

Return ONLY the JSON.
"""

    def _build_contextual_french_prompt(self, user_profile, job_profile, rag_advice):
        """Prompt français avec reformulation contextuelle"""
            # 🔥 AJOUT: Vérifier si le CV original contient des projets
        has_projects = bool(user_profile.get('projets') and len(user_profile.get('projets', [])) > 0)
        projects_warning = ""
    
        if has_projects:
         projects_warning = """
⚠️ ATTENTION CRITIQUE: Ce CV contient des PROJETS existants.
Les projets sont OBLIGATOIRES dans le JSON de sortie.
NE JAMAIS omettre ou supprimer les projets du CV original.
"""
        return f"""
# GUIDE D'AMÉLIORATION CONTEXTUELLE:
{rag_advice}
{projects_warning}

# TÂCHE: Reformulation contextuelle du CV
Reformulez le CV avec le vocabulaire de la description de poste.

# DONNÉES ORIGINALES:
- competences_techniques: {json.dumps(user_profile.get('competences_techniques', []))}
- competences_soft: {json.dumps(user_profile.get('competences_soft', []))}
- experiences: {json.dumps(user_profile.get('experiences', [])[:3])}
- projets: {json.dumps(user_profile.get('projets', []))}
- formation: "{user_profile.get('formation', '')}"
- objectif_professionnel: "{user_profile.get('objectif_professionnel', '')}"

# CONTEXTE DU POSTE:
- Poste: {job_profile.get('objectif_professionnel', '')}
- Compétences clés: {', '.join(job_profile.get('competences_techniques', [])[:8])}
- Mots-clés: {self._extract_key_terms(job_profile.get('raw_job_text', ''))}

# STRATÉGIE:
1. COMPÉTENCES CACHÉES: Analyser les expériences pour extraire les compétences techniques implicites correspondant au poste
2. INFÉRENCE INTELLIGENTE: Si une expérience implique une compétence demandée par le job, l'ajouter aux compétences_techniques
3. ALIGNEMENT VOCABULAIRE: Utiliser la terminologie de la description de poste
4. REFORMULATION EXPÉRIENCES: Rephrasez les expériences avec le vocabulaire du poste
5. MAPPING INTELLIGENT: Relier intelligemment le travail réel du candidat aux exigences du poste
6. PRÉSERVER LA VÉRITÉ: N'ajouter que les compétences réellement impliquées par les expériences
7. NE PAS INVENTER: Jamais de compétences complètement fictives

# EXEMPLES D'EXTRACTION DE COMPÉTENCES CACHÉES:
- Expérience: "Développement de modèles LLM pour génération de contenu marketing"
  Poste demande: "Expérience en IA générative"
  → Ajouter: "IA générative" aux compétences_techniques ✅

- Expérience: "Analyse de données de vente pour prédire les tendances"
  Poste demande: "Modélisation prédictive des ventes"
  → Ajouter: "Modélisation prédictive" aux compétences_techniques ✅

- Expérience: "Manipulation de données avec Python"
  Poste demande: "Analyse de données avec Pandas"
  → Ajouter: "Pandas" aux compétences_techniques ✅

# EXEMPLES DE REFORMULATION:
- Original: "Application pour anticiper les ventes"
  Terme du poste: "Modèles prédictifs de ventes"
  → Rephrasez: "Construction de modèles prédictifs de ventes" ✅

- Original: "Travail avec les données clients"
  Terme du poste: "Analytics clients"
  → Rephrasez: "Réalisation d'analytics clients" ✅

# FORMAT (mêmes champs exactement):
{{
    "competences_techniques": ["compétences reformulées"],
    "competences_soft": ["soft skills reformulées"],
    "experiences": ["expériences reformulées"],
    "projets": ["projets reformulés"],
    "formation": "formation reformulée",
    "objectif_professionnel": "objectif reformulé",
    "localisation": "{user_profile.get('localisation', '')}",
    "nom_complet": "{user_profile.get('nom_complet', '')}",
    "telephone": "{user_profile.get('telephone', '')}",
    "email": "{user_profile.get('email', '')}",
    "job_title": "{user_profile.get('job_title', '')}"
}}

Retournez UNIQUEMENT le JSON.
"""

    def _parse_json_response(self, response):
        """Parse JSON - SANS FALLBACK"""
        try:
            response = response.strip()
            response = re.sub(r'```json\s*', '', response)
            response = re.sub(r'```\s*', '', response)
            response = re.sub(r'^[^{]*', '', response)
            response = re.sub(r'}[^}]*$', '}', response)

            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                enhanced_cv = json.loads(json_str)

                required = ['competences_techniques', 'competences_soft', 'experiences',
                           'formation', 'objectif_professionnel']

                if all(key in enhanced_cv for key in required):
                    return enhanced_cv

            return None

        except json.JSONDecodeError as e:
            print(f"⚠️ JSON invalide: {e}")
            return None
        except Exception as e:
            print(f"⚠️ Erreur parsing: {e}")
            return None

    def _validate_ethical_enhancement(self, original, enhanced):
        """Validation éthique - INFO SEULEMENT"""
        try:
            original_skills = set([s.lower() for s in original.get('competences_techniques', [])])
            enhanced_skills = enhanced.get('competences_techniques', [])

            if not isinstance(enhanced_skills, list):
                return False

            invented = []
            for enhanced_skill in enhanced_skills:
                if not isinstance(enhanced_skill, str):
                    continue

                enhanced_lower = enhanced_skill.lower()
                is_valid = False

                for original_skill in original_skills:
                    original_words = set(original_skill.lower().split())
                    enhanced_words = set(enhanced_lower.split())

                    common = original_words.intersection(enhanced_words)
                    meaningful_common = [w for w in common if len(w) > 3]

                    if meaningful_common:
                        is_valid = True
                        break

                if not is_valid and "en cours" not in enhanced_lower and "learning" not in enhanced_lower:
                    invented.append(enhanced_skill)

            if invented:
                print(f"🚨 COMPÉTENCES INVENTÉES: {invented}")
                return False

            return True

        except Exception as e:
            print(f"⚠️ Erreur validation: {e}")
            return True

# ============================================
# CELLULE 2: CLASSE LaTeXCVGenerator
# ============================================

class LaTeXCVGenerator:
    def __init__(self, groq_processor):
        self.groq = groq_processor
        self.template = self._load_template()

    def _load_template(self):
        """Retourne le template LaTeX de base"""
        return r"""
%-----------------------------------------------------------------------------------------------------------------------------------------------%
%	The MIT License (MIT)
%
%	Copyright (c) 2021 Jitin Nair
%
%	Permission is hereby granted, free of charge, to any person obtaining a copy
%	of this software and associated documentation files (the "Software"), to deal
%	in the Software without restriction, including without limitation the rights
%	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
%	copies of the Software, and to permit persons to whom the Software is
%	furnished to do so, subject to the following conditions:
%
%	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
%	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
%	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
%	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
%	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
%	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
%	THE SOFTWARE.
%
%
%-----------------------------------------------------------------------------------------------------------------------------------------------%

%----------------------------------------------------------------------------------------
%	DOCUMENT DEFINITION
%----------------------------------------------------------------------------------------

\documentclass[a4paper,12pt]{article}

%----------------------------------------------------------------------------------------
%	PACKAGES
%----------------------------------------------------------------------------------------
\usepackage{url}
\usepackage{parskip}

\RequirePackage{color}
\RequirePackage{graphicx}
\usepackage[usenames,dvipsnames]{xcolor}
\usepackage[scale=0.9]{geometry}

\usepackage{tabularx}
\usepackage{enumitem}

\newcolumntype{C}{>{\centering\arraybackslash}X}

\usepackage{supertabular}
\usepackage{tabularx}
\newlength{\fullcollw}
\setlength{\fullcollw}{0.47\textwidth}

\usepackage{titlesec}
\usepackage{multicol}
\usepackage{multirow}

\titleformat{\section}{\Large\scshape\raggedright}{}{0em}{}[\titlerule]
\titlespacing{\section}{0pt}{10pt}{10pt}

\usepackage[style=authoryear,sorting=ynt, maxbibnames=2]{biblatex}

\usepackage[unicode, draft=false]{hyperref}
\definecolor{linkcolour}{rgb}{0,0.2,0.6}
\hypersetup{colorlinks,breaklinks,urlcolor=linkcolour,linkcolor=linkcolour}
\addbibresource{citations.bib}
\setlength\bibitemsep{1em}

\usepackage{fontawesome5}

\newenvironment{jobshort}[2]
    {
    \begin{tabularx}{\linewidth}{@{}l X r@{}}
    \textbf{#1} & \hfill &  #2 \\[3.75pt]
    \end{tabularx}
    }
    {
    }

\newenvironment{joblong}[2]
    {
    \begin{tabularx}{\linewidth}{@{}l X r@{}}
    \textbf{#1} & \hfill &  #2 \\[3.75pt]
    \end{tabularx}
    \begin{minipage}[t]{\linewidth}
    \begin{itemize}[nosep,after=\strut, leftmargin=1em, itemsep=3pt,label=--]
    }
    {
    \end{itemize}
    \end{minipage}
    }

%----------------------------------------------------------------------------------------
%	BEGIN DOCUMENT
%----------------------------------------------------------------------------------------
\begin{document}

\pagestyle{empty}

%----------------------------------------------------------------------------------------
%	TITLE
%----------------------------------------------------------------------------------------

{TITLE_SECTION}

%----------------------------------------------------------------------------------------
% EXPERIENCE SECTIONS
%----------------------------------------------------------------------------------------

{SUMMARY_SECTION}

{EXPERIENCE_SECTION}

{PROJECTS_SECTION}

{EDUCATION_SECTION}

{SKILLS_SECTION}

\vfill


\end{document}
"""

    def generate_latex_cv(self, enhanced_cv_json, job_info=None):
        """Génère le CV LaTeX - VERSION CORRIGÉE"""
        print("📝 Génération du CV LaTeX...")

        # Valider et nettoyer le CV d'abord
        clean_cv = self._validate_and_clean_cv(enhanced_cv_json)

        # Générer chaque section
        title_section = self._generate_title_section(clean_cv)
        summary_section = self._generate_summary_section(clean_cv)
        experience_section = self._generate_experience_section(clean_cv)
        projects_section = self._generate_projects_section(clean_cv)
        education_section = self._generate_education_section(clean_cv)
        skills_section = self._generate_skills_section(clean_cv)

        # Remplir le template
        latex_cv = self.template.replace("{TITLE_SECTION}", title_section)
        latex_cv = latex_cv.replace("{SUMMARY_SECTION}", summary_section)
        latex_cv = latex_cv.replace("{EXPERIENCE_SECTION}", experience_section)
        latex_cv = latex_cv.replace("{PROJECTS_SECTION}", projects_section)
        latex_cv = latex_cv.replace("{EDUCATION_SECTION}", education_section)
        latex_cv = latex_cv.replace("{SKILLS_SECTION}", skills_section)

        print("✅ CV LaTeX généré")
        return latex_cv

    def _detect_cv_language(self, cv_data):
        """Détecte la langue du CV"""
        sample_text = ""
        if cv_data.get('objectif_professionnel'):
            sample_text += cv_data.get('objectif_professionnel', '')
        if cv_data.get('experiences'):
            exps = cv_data.get('experiences', [])[:2]
            for exp in exps:
                if isinstance(exp, dict):
                    sample_text += ' ' + exp.get('description', '')
                else:
                    sample_text += ' ' + str(exp)

        french_indicators = ['chez', 'compétence', 'expérience', 'formation', 'projet']
        english_indicators = ['at', 'skill', 'experience', 'education', 'project']

        sample_lower = sample_text.lower()
        french_count = sum(1 for word in french_indicators if word in sample_lower)
        english_count = sum(1 for word in english_indicators if word in sample_lower)

        return "french" if french_count > english_count else "english"

    def _validate_and_clean_cv(self, cv_data):
        """Valide et nettoie le CV avant génération LaTeX"""
        print("🧹 Nettoyage du CV pour LaTeX...")

        clean_cv = {
            'nom_complet': '',
            'email': '',
            'telephone': '',
            'job_title': '',
            'objectif_professionnel': '',
            'competences_techniques': [],
            'competences_soft': [],
            'experiences': [],
            'projets': [],
            'formation': ''
        }

        for field in ['nom_complet', 'email', 'telephone', 'job_title', 'objectif_professionnel']:
            value = cv_data.get(field, '')
            if value and isinstance(value, str):
                clean_cv[field] = value.strip()

        tech_skills = cv_data.get('competences_techniques', [])
        if isinstance(tech_skills, list):
            clean_cv['competences_techniques'] = [
                str(s).strip()
                for s in tech_skills
                if s and isinstance(s, str) and not s.startswith('[')
            ][:15]

        soft_skills = cv_data.get('competences_soft', [])
        if isinstance(soft_skills, list):
            clean_cv['competences_soft'] = [
                str(s).strip()
                for s in soft_skills
                if s and isinstance(s, str) and not s.startswith('[')
            ][:10]

        experiences = cv_data.get('experiences', [])
        if isinstance(experiences, list):
            clean_cv['experiences'] = self._parse_experiences(experiences)[:5]

        projets = cv_data.get('projets', [])
        if isinstance(projets, list):
            clean_cv['projets'] = self._parse_projets(projets)[:3]

        formation = cv_data.get('formation', '')
        clean_cv['formation'] = self._parse_formation(formation)

        print(f"   ✅ {len(clean_cv['competences_techniques'])} compétences techniques")
        print(f"   ✅ {len(clean_cv['experiences'])} expériences")
        print(f"   ✅ {len(clean_cv['projets'])} projets")
        print(f"   ✅ Formation parsée")

        return clean_cv

    def _parse_experiences(self, experiences):
        """Parse les expériences et extrait les informations structurées"""
        import ast
        parsed_experiences = []

        for exp in experiences:
            if not exp or (isinstance(exp, str) and len(str(exp).strip()) < 10):
                continue

            if isinstance(exp, str):
                if exp.strip().startswith('{'):
                    try:
                        exp_dict = ast.literal_eval(exp)
                        parsed_experiences.append(exp_dict)
                    except:
                        parsed_experiences.append({'description': exp})
                else:
                    parsed_exp = self._extract_experience_info(exp)
                    parsed_experiences.append(parsed_exp)
            elif isinstance(exp, dict):
                parsed_experiences.append(exp)

        return parsed_experiences

    def _extract_experience_info(self, exp_text):
        """Extrait les informations d'une expérience depuis du texte brut"""
        exp_dict = {'description': exp_text}

        if ' chez ' in exp_text or ' at ' in exp_text:
            parts = exp_text.split(' chez ' if ' chez ' in exp_text else ' at ')
            if len(parts) >= 2:
                exp_dict['poste'] = parts[0].strip()
                entreprise_part = parts[1].split(',')[0].strip()
                exp_dict['entreprise'] = entreprise_part

        return exp_dict

    def _parse_projets(self, projets):
        """Parse les projets"""
        import ast
        parsed_projets = []

        for projet in projets:
            if not projet or (isinstance(projet, str) and len(str(projet).strip()) < 10):
                continue

            if isinstance(projet, str):
                if projet.strip().startswith('{'):
                    try:
                        projet_dict = ast.literal_eval(projet)
                        parsed_projets.append(projet_dict)
                    except:
                        parsed_projets.append({'description': projet})
                else:
                    parsed_projets.append({'description': projet})
            elif isinstance(projet, dict):
                parsed_projets.append(projet)

        return parsed_projets

    def _parse_formation(self, formation):
        """Parse la formation en liste de dicts"""
        import ast

        if not formation:
            return []

        if isinstance(formation, list):
            result = []
            for form in formation:
                if isinstance(form, dict):
                    result.append(form)
                elif isinstance(form, str):
                    if form.strip().startswith('{'):
                        try:
                            form_dict = ast.literal_eval(form)
                            result.append(form_dict)
                        except:
                            result.append({'diplome': form})
                    else:
                        result.append({'diplome': form})
            return result

        if isinstance(formation, str):
            if formation.strip().startswith('['):
                try:
                    formations_list = ast.literal_eval(formation)
                    if isinstance(formations_list, list):
                        return formations_list
                except Exception as e:
                    print(f"⚠️ Erreur parsing liste formation: {e}")

            if formation.strip().startswith('{'):
                try:
                    formation_dict = ast.literal_eval(formation)
                    return [formation_dict]
                except:
                    pass

            separators = [' et ', ', ']
            formations = [formation]

            for sep in separators:
                if sep in formation:
                    formations = formation.split(sep)
                    break

            result = []
            for form in formations:
                form = form.strip()
                if len(form) > 5:
                    form_dict = {'diplome': form}

                    for keyword in [' de l\'', ' de ', ' à ', ' - ', ', ']:
                        if keyword in form:
                            parts = form.split(keyword, 1)
                            form_dict['diplome'] = parts[0].strip()
                            form_dict['etablissement'] = parts[1].strip()
                            break

                    result.append(form_dict)

            return result

        return []

    def _escape_latex(self, text):
        """Échappe les caractères spéciaux LaTeX"""
        if not text:
            return ""
        text = str(text)
        replacements = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\^{}',
            '\\': r'\textbackslash{}',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _generate_title_section(self, cv_data):
        """Génère la section titre"""
        name = self._escape_latex(cv_data.get('nom_complet', 'Your Name'))
        email = cv_data.get('email', 'email@email.com')
        phone = cv_data.get('telephone', '+00.00.000.000')

        return rf"""
\begin{{tabularx}}{{\linewidth}}{{@{{}} C @{{}}}}
\Huge{{{name}}} \\[7.5pt]
\href{{mailto:{email}}}{{\raisebox{{-0.05\height}}\faEnvelope \ {email}}} \ $|$ \
\href{{tel:{phone}}}{{\raisebox{{-0.05\height}}\faMobile \ {phone}}} \\
\end{{tabularx}}
"""

    def _generate_summary_section(self, cv_data):
        """Génère la section Summary/Résumé"""
        objective = self._escape_latex(cv_data.get('objectif_professionnel', ''))
        if not objective:
            return ""

        is_french = self._detect_cv_language(cv_data) == "french"
        section_title = "Résumé" if is_french else "Summary"

        return rf"""
\section{{{section_title}}}
{objective}
"""

    def _generate_experience_section(self, cv_data):
        """Génère la section Work Experience - SANS RÉPÉTITION DU TITRE"""
        experiences = cv_data.get('experiences', [])
        if not experiences:
            return ""

        is_french = self._detect_cv_language(cv_data) == "french"
        section_title = "Expérience Professionnelle" if is_french else "Work Experience"

        section = rf"\section{{{section_title}}}" + "\n\n"

        for exp in experiences[:5]:
            if isinstance(exp, dict):
                entreprise = self._escape_latex(exp.get('entreprise', ''))
                poste = self._escape_latex(exp.get('poste', ''))
                date_debut = exp.get('date_debut', '')
                date_fin = exp.get('date_fin', '')

                # Gestion de "présent"
                if date_fin and 'pr' in str(date_fin).lower():
                    date_fin = 'Présent' if is_french else 'Present'

                dates = f"{date_debut} - {date_fin}" if date_debut and date_fin else ""

                # Titre: Poste - Entreprise (SANS répétition dans les tâches)
                if poste and entreprise:
                    title = f"{poste} - {entreprise}"
                elif poste:
                    title = poste
                elif entreprise:
                    title = entreprise
                else:
                    title = "Experience"

                taches = exp.get('taches', [])
                description = exp.get('description', '')

                # Si pas de tâches, parser la description
                if not taches and description:
                    taches = self._parse_description_to_bullets(description)

                # Générer avec bullet points SANS répéter le titre
                if taches and isinstance(taches, list) and len(taches) > 0:
                    section += rf"\begin{{joblong}}{{{title}}}{{{dates}}}" + "\n"
                    
                    for tache in taches[:5]:
                        tache_str = str(tache).strip()
                        
                        # ÉVITER LA RÉPÉTITION: Ne pas inclure si la tâche répète exactement le poste/entreprise
                        tache_lower = tache_str.lower()
                        should_skip = False
                        
                        if poste and entreprise:
                            # Vérifier si la tâche est juste une répétition du titre
                            if (poste.lower() in tache_lower and entreprise.lower() in tache_lower and 
                                len(tache_str) < len(f"{poste} chez {entreprise}") + 20):
                                should_skip = True
                        
                        if not should_skip:
                            tache_clean = self._escape_latex(tache_str)
                            section += f"    \\item {tache_clean}\n"
                    
                    section += r"\end{joblong}" + "\n\n"
                else:
                    # Format court si pas de tâches
                    desc_clean = self._escape_latex(str(description)[:200]) if description else ""
                    section += rf"\begin{{jobshort}}{{{title}}}{{{dates}}}" + "\n"
                    if desc_clean:
                        section += desc_clean + "\n"
                    section += r"\end{jobshort}" + "\n\n"
                    
            else:
                # Gérer les expériences en format string
                exp_str = str(exp)
                title, dates, bullets = self._parse_raw_experience(exp_str)

                if bullets and len(bullets) > 0:
                    section += rf"\begin{{joblong}}{{{self._escape_latex(title)}}}{{{dates}}}" + "\n"
                    for bullet in bullets[:5]:
                        bullet_clean = self._escape_latex(bullet)
                        # Éviter répétition du titre dans les bullets
                        if not (title.lower() in bullet.lower() and len(bullet) < len(title) + 20):
                            section += f"    \\item {bullet_clean}\n"
                    section += r"\end{joblong}" + "\n\n"
                else:
                    exp_clean = self._escape_latex(exp_str[:200])
                    section += rf"\begin{{jobshort}}{{Experience}}{{{dates}}}" + "\n"
                    section += exp_clean + "\n"
                    section += r"\end{jobshort}" + "\n\n"

        return section

    def _parse_description_to_bullets(self, description):
        """Parse une description longue en bullet points"""
        if not description or len(description) < 50:
            return []

        bullets = []
        separators = [", où j'ai ", ", et j'ai ", ". J'ai ", " et j'ai ", " et ", ", j'ai "]

        parts = [description]
        for sep in separators:
            new_parts = []
            for part in parts:
                if sep in part:
                    new_parts.extend(part.split(sep))
                else:
                    new_parts.append(part)
            parts = new_parts

        for part in parts:
            part = part.strip()
            for prefix in ['où ', 'et ', ', ']:
                if part.startswith(prefix):
                    part = part[len(prefix):].strip()

            if part and len(part) > 10:
                part = part[0].upper() + part[1:]
                bullets.append(part)

        if len(bullets) < 2:
            bullets = self._parse_by_sentence(description)

        return bullets[:5]

    def _parse_by_sentence(self, description):
        """Parse par phrases"""
        import re
        sentences = re.split(r'[.;]', description)

        bullets = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:
                sentence = sentence[0].upper() + sentence[1:] if sentence else sentence
                bullets.append(sentence)

        return bullets

    def _parse_raw_experience(self, exp_text):
        """Parse une expérience brute"""
        title = "Experience"
        dates = ""
        bullets = []

        if ' chez ' in exp_text:
            parts = exp_text.split(' chez ', 1)
            title = parts[0].strip()
            remaining = parts[1]

            if ',' in remaining:
                entreprise = remaining.split(',')[0].strip()
                title = f"{title} - {entreprise}"
                description = ','.join(remaining.split(',')[1:]).strip()
            else:
                description = remaining
        elif ' at ' in exp_text:
            parts = exp_text.split(' at ', 1)
            title = parts[0].strip()
            remaining = parts[1]

            if ',' in remaining:
                company = remaining.split(',')[0].strip()
                title = f"{title} - {company}"
                description = ','.join(remaining.split(',')[1:]).strip()
            else:
                description = remaining
        else:
            description = exp_text

        bullets = self._parse_description_to_bullets(description)

        return title, dates, bullets

    def _generate_projects_section(self, cv_data):
        """Génère la section Projets - FORMAT BULLET POINTS AVEC DATES"""
        projets = cv_data.get('projets', [])
        if not projets:
            return ""

        is_french = self._detect_cv_language(cv_data) == "french"
        section_title = "Projets" if is_french else "Projects"

        section = rf"\section{{{section_title}}}" + "\n\n"

        for projet in projets[:3]:
            if isinstance(projet, dict):
                titre = self._escape_latex(projet.get('titre', '') or projet.get('nom', 'Project'))
                date = projet.get('date', '')
                description = projet.get('description', '')
                technologies = projet.get('technologies', [])
                lien = projet.get('lien', '')

                # Générer la ligne de titre avec date
                section += rf"\begin{{joblong}}{{{titre}}}{{{date}}}" + "\n"
                
                # Description comme bullet point
                if description:
                    desc_clean = self._escape_latex(str(description))
                    section += f"    \\item {desc_clean}\n"

                # Technologies comme bullet point
                if technologies and isinstance(technologies, list):
                    tech_str = ", ".join([str(t) for t in technologies])
                    tech_clean = self._escape_latex(tech_str)
                    tech_label = "Technologies" if not is_french else "Technologies"
                    section += f"    \\item {tech_label}: {tech_clean}\n"
                
                # Lien comme bullet point
                if lien and lien != 'Link to Demo':
                    section += f"    \\item \\href{{{lien}}}{{Lien du projet}}\n"

                section += r"\end{joblong}" + "\n\n"
            else:
                # Si c'est du texte brut, extraire ce qu'on peut
                projet_str = str(projet)
                projet_clean = self._escape_latex(projet_str[:200])
                section += rf"\begin{{jobshort}}{{Project}}{{}}" + "\n"
                section += projet_clean + "\n"
                section += r"\end{jobshort}" + "\n\n"

        return section

    def _generate_education_section(self, cv_data):
        """Génère la section Education - FORMAT TABLEAU"""
        formations = cv_data.get('formation', [])
        if not formations:
            return ""

        is_french = self._detect_cv_language(cv_data) == "french"
        section_title = "Formation" if is_french else "Education"

        section = rf"\section{{{section_title}}}" + "\n"
        section += r"\begin{tabularx}{\linewidth}{@{}l X@{}}" + "\n"

        if isinstance(formations, list):
            for form in formations:
                if isinstance(form, dict):
                    diplome = self._escape_latex(form.get('diplome', ''))
                    etablissement = self._escape_latex(form.get('etablissement', ''))
                    date_debut = form.get('date_debut', '')
                    date_fin = form.get('date_fin', '')
                    gpa = form.get('gpa', '')

                    # Gestion des dates
                    if date_fin and 'pr' in str(date_fin).lower():
                        date_fin = 'présent' if is_french else 'present'
                    
                    dates = f"{date_debut} - {date_fin}" if date_debut and date_fin else date_fin or date_debut or ""
                    
                    # Gestion du GPA
                    gpa_str = ""
                    if gpa:
                        gpa_str = rf"\hfill \normalsize (GPA: {gpa})"

                    # Format: dates & Diplôme at Établissement
                    at_word = "à" if is_french else "at"
                    if etablissement:
                        line = rf"{dates} & {diplome} {at_word} \textbf{{{etablissement}}} \\"
                    else:
                        line = rf"{dates} & {diplome}  \\"
                    
                    section += line + "\n"
                else:
                    # Format simple si ce n'est pas un dict
                    form_clean = self._escape_latex(str(form))
                    section += rf"& {form_clean} \\" + "\n"
        else:
            form_clean = self._escape_latex(str(formations))
            section += rf"& {form_clean} \\" + "\n"

        section += r"\end{tabularx}" + "\n"
        return section

    def _generate_skills_section(self, cv_data):
        """Génère la section Skills"""
        tech_skills = cv_data.get('competences_techniques', [])
        soft_skills = cv_data.get('competences_soft', [])

        if not tech_skills and not soft_skills:
            return ""

        is_french = self._detect_cv_language(cv_data) == "french"
        section_title = "Compétences" if is_french else "Skills"
        tech_label = "Compétences Techniques" if is_french else "Technical Skills"
        soft_label = "Compétences Comportementales" if is_french else "Soft Skills"

        tech_str = self._escape_latex(', '.join(tech_skills[:15]))
        soft_str = self._escape_latex(', '.join(soft_skills[:10]))

        section = rf"\section{{{section_title}}}" + "\n"
        section += r"\begin{tabularx}{\linewidth}{@{}l X@{}}" + "\n"

        if tech_str:
            section += rf"{tech_label} &  \normalsize{{{tech_str}}}\\" + "\n"

        if soft_str:
            section += rf"{soft_label}  &  \normalsize{{{soft_str}}}\\" + "\n"

        section += r"\end{tabularx}" + "\n"

        return section

    def save_latex_to_file(self, latex_content, filename="enhanced_cv.tex"):
        """Sauvegarde le fichier LaTeX"""
        try:
            # with open(filename, 'w', encoding='utf-8') as f:
            #   f.write(latex_content)
            print(f"💾 CV LaTeX sauvegardé: {filename}")
            return filename
        except Exception as e:
            print(f"❌ Erreur sauvegarde LaTeX: {e}")
            return None

    def compile_to_pdf(self, latex_filename):
     """Compile le LaTeX en PDF - VERSION CORRECTE"""
     print(f"🔧 Compilation de {latex_filename} en PDF...")
    
     if not os.path.exists(latex_filename):
         print(f"❌ Fichier .tex introuvable: {latex_filename}")
         return None

     try:
        # Vérifier pdflatex
         result = subprocess.run(['pdflatex', '--version'], capture_output=True, text=True)
         if result.returncode != 0:
             print("❌ pdflatex non trouvé")
             return None
        
         output_dir = os.path.dirname(latex_filename) or os.getcwd()
        
        # Compiler 2 fois (pour les références)
         for i in range(2):
             print(f"🔄 Compilation {i+1}/2...")
             result = subprocess.run(
                 ["pdflatex", "-interaction=nonstopmode", "-output-directory", output_dir, latex_filename],
                 capture_output=True, text=True, timeout=500
             )
            
             if result.returncode != 0:
                 print(f"❌ Erreur compilation {i+1}: {result.stderr[-500:]}")
                 return None

         pdf_filename = latex_filename.replace('.tex', '.pdf')
        
         if os.path.exists(pdf_filename):
             print(f"✅ PDF généré avec succès: {pdf_filename}")
             return pdf_filename
         else:
             print("❌ PDF non généré - vérifiez les erreurs LaTeX")
             if result.stderr:
                 print("Dernières erreurs:", result.stderr[-500:])
             return None

     except Exception as e:
         print(f"❌ Erreur compilation: {e}")
         return None

class EnhancedCareerAgent:
    def __init__(self, config):
        self.config = config
        print("🔄 Initialisation de l'Agent IA Amélioré avec Groq...")

        # Initialisation des composants
        self.groq_processor = GroqProcessor(config.GROQ_API_KEY)
        self.pdf_parser = PDFCVParser(self.groq_processor)
        self.latex_generator = LaTeXCVGenerator(self.groq_processor)
        self.profile_analyzer = ProfileAnalyzer(self.groq_processor)
        self.job_analyzer = JobAnalyzer(self.groq_processor)
        self.matcher = IntelligentJobMatcher(self.groq_processor)
        self.scraper = EnhancedHybridJobScraper(config.APIFY_API_KEY, similarity_threshold=config.similarity_threshold)
        self.enhancer = EthicalRAGCVEnhancer(self.groq_processor)
        print("✅ Agent IA Amélioré avec Groq initialisé")

    def process_user_request(self, cv_data, user_preferences):
        """
        Workflow complet pour un CV déjà structuré (non-PDF)
        """
        print("\n" + "="*60)
        print("🤖 AGENT GROQ 70B - WORKFLOW JSON → Enhancement")
        print("="*60)

        # Étape 1: Analyse du CV
        print("\n1. 🔍 ANALYSE DU CV...")
        try:
            user_profile = self.profile_analyzer.analyze_cv(cv_data)
            if not user_profile:
                return {"error": "ERREUR: Impossible d'analyser le CV"}
        except Exception as e:
            return {"error": f"ERREUR analyse CV: {str(e)}"}

        # Étape 2: Recherche jobs
        print("\n2. 🔎 RECHERCHE JOBS...")
        try:
            jobs = self.scraper.search_jobs(user_profile, user_preferences or {})
            if not jobs:
                return {"error": "ERREUR: Aucun job trouvé"}
        except Exception as e:
            return {"error": f"ERREUR recherche jobs: {str(e)}"}

        # Étape 3: Matching
        print("\n3. 🎯 MATCHING...")
        try:
            best_match = self.matcher.find_best_match(user_profile, jobs)
            if not best_match:
                return {"error": "ERREUR: Aucun match trouvé"}
        except Exception as e:
            return {"error": f"ERREUR matching: {str(e)}"}

        # Étape 4: Enhancement
        print("\n4. ✨ ENHANCEMENT...")
        try:
            job_profile = self.job_analyzer.analyze_job(best_match['job'])
            enhanced_cv = self.enhancer.enhance_cv_ethically(user_profile, job_profile)
        except Exception as e:
            print(f"❌ Erreur enhancement: {e}")
            enhanced_cv = user_profile

        return {
            "success": True,
            "llm_engine": "Groq + Llama 3.3 70B",
            "candidate_info": {
                "nom_complet": user_profile.get('nom_complet'),
                "email": user_profile.get('email'),
                "telephone": user_profile.get('telephone')
            },
            "user_profile": user_profile,
            "recommended_job": best_match['job'],
            "enhanced_cv_json": enhanced_cv,
            "match_score": best_match.get('contextual_score', 0),
            "contextual_score": best_match.get('contextual_score', 0),
            "final_matching_score": best_match.get('final_score', 0),
            "match_reasoning": best_match.get('reasoning', ''),
            "matching_strengths": best_match.get('score_details', {}).get('reasoning', ''),
            "source_used": best_match['job'].get('source', 'Unknown'),
            "total_jobs_found": len(jobs),
            "job_link": best_match['job'].get('url', ''),
            "timestamp": datetime.now().isoformat()
        }

    def process_pdf_cv_orchestrated(self, pdf_path, user_preferences=None):
     """
     Workflow complet avec orchestration LangGraph
     """
     print("\n" + "="*60)
     print("🤖 AGENT GROQ 70B - WORKFLOW ORCHESTRÉ")
     print("="*60)

     # Étape 0: Parser le PDF
     print("\n0. 📄 EXTRACTION PDF...")
     try:
         cv_data = self.pdf_parser.process_pdf_cv(pdf_path)
         if not cv_data:
             return {"error": "ERREUR: Impossible d'extraire le CV du PDF"}
     except Exception as e:
         return {"error": f"ERREUR extraction PDF: {str(e)}"}

    # Étape 1: Analyse du CV
     print("\n1. 🔍 ANALYSE DU CV...")
     try:
         user_profile = self.profile_analyzer.analyze_cv(cv_data)
         if not user_profile:
             return {"error": "ERREUR: Impossible d'analyser le CV"}
     except Exception as e:
         return {"error": f"ERREUR analyse CV: {str(e)}"}

    # Étape 2: Recherche jobs
     print("\n2. 🔎 RECHERCHE JOBS...")
     try:
         jobs = self.scraper.search_jobs(user_profile, user_preferences or {})
         if not jobs:
             return {"error": "ERREUR: Aucun job trouvé"}
     except Exception as e:
         return {"error": f"ERREUR recherche jobs: {str(e)}"}

    # Étape 3: Matching
     print("\n3. 🎯 MATCHING...")
     try:
         best_match = self.matcher.find_best_match(user_profile, jobs)
         if not best_match:
             return {"error": "ERREUR: Aucun match trouvé"}
        
         job_profile = self.job_analyzer.analyze_job(best_match['job'])
     except Exception as e:
         return {"error": f"ERREUR matching: {str(e)}"}

    # Étape 4-6: ORCHESTRATION avec LangGraph
     print("\n4-6. 🎭 ORCHESTRATION (Enhancement + LaTeX + Validation)...")
     orchestrator = CVOrchestrator(self)
    
     result = orchestrator.run_orchestrated_generation(
         user_profile=user_profile,
         job_profile=job_profile,
         max_retries=2
     )

     if result['success']:
         return {
             "success": True,
             "user_profile": user_profile,
             "recommended_job": best_match['job'],
             "enhanced_cv_json": result['enhanced_cv'],
             "pdf_filename": result['pdf_path'],
             "match_score": best_match.get('final_score', 0),
             "total_retries": result['total_retries'],
             "timestamp": datetime.now().isoformat()
         }
     else:
         return {
             "success": False,
             "error": f"Orchestration échouée: {result['status']}",
             "errors": result.get('errors', []),
             "total_retries": result['total_retries'],
             "partial_data": {
                 "user_profile": user_profile,
                 "recommended_job": best_match['job'],
                 "partial_cv": result.get('partial_cv'),
                 "partial_pdf": result.get('partial_pdf')
             }
         }

    def get_agent_info(self):
        """Retourne les informations sur l'agent Groq"""
        return {
            "agent_name": "EnhancedCareerAgent",
            "llm_engine": "Groq + Llama 3.3 70B Versatile",
            "features": [
                "Analyse contextuelle des CV",
                "Matching contextuel pur avec Groq 70B",
                "Enhancement éthique avec RAG",
                "Scraping hybride multi-sources"
            ],
            "sources_supported": ["LinkedIn"],
            "matching_approach": "Contextuel pur (LLM seulement)",
            "similarity_threshold": self.config.similarity_threshold,
            "components": [
                "ProfileAnalyzer",
                "JobAnalyzer",
                "IntelligentJobMatcher (contextuel pur)",
                "EnhancedHybridJobScraper",
                "EthicalRAGCVEnhancer"
            ]
        }


class CVGenerationState(TypedDict):
    """État pour l'orchestration LangGraph"""
    user_profile: dict
    job_profile: dict
    enhanced_cv: dict
    latex_content: str
    pdf_path: str
    validation_errors: list
    retry_count: int
    max_retries: int
    status: Literal["pending", "json_invalid", "latex_invalid", "pdf_multipage", "success", "failed"]


class CVOrchestrator:
    """
    Orchestrateur avec LangGraph pour validation et retry intelligent
    """
    def __init__(self, agent: EnhancedCareerAgent):
        self.agent = agent
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Construit le graph LangGraph"""
        workflow = StateGraph(CVGenerationState)
        
        # Nodes
        workflow.add_node("enhance_cv", self._enhance_cv_node)
        workflow.add_node("validate_json", self._validate_json_node)
        workflow.add_node("generate_latex", self._generate_latex_node)
        workflow.add_node("validate_latex", self._validate_latex_node)
        workflow.add_node("compile_pdf", self._compile_pdf_node)
        workflow.add_node("validate_pdf", self._validate_pdf_node)
        workflow.add_node("retry_enhancement", self._retry_enhancement_node)
        workflow.add_node("retry_latex", self._retry_latex_node)
        
        # Entry point
        workflow.set_entry_point("enhance_cv")
        
        # Edges conditionnels
        workflow.add_conditional_edges(
            "enhance_cv",
            self._route_after_enhancement,
            {
                "validate": "validate_json",
                "failed": "retry_enhancement"
            }
        )
        
        workflow.add_conditional_edges(
            "validate_json",
            self._route_after_json_validation,
            {
                "valid": "generate_latex",
                "retry": "retry_enhancement",
                "failed": "retry_enhancement"
            }
        )
        
        workflow.add_edge("retry_enhancement", "enhance_cv")
        
        workflow.add_conditional_edges(
            "generate_latex",
            self._route_after_latex_generation,
            {
                "validate": "validate_latex",
                "failed": "retry_latex"
            }
        )
        
        workflow.add_conditional_edges(
            "validate_latex",
            self._route_after_latex_validation,
            {
                "valid": "compile_pdf",
                "retry": "retry_latex",
                "failed": "retry_latex"
            }
        )
        
        workflow.add_edge("retry_latex", "generate_latex")
        
        workflow.add_conditional_edges(
            "compile_pdf",
            self._route_after_compilation,
            {
                "validate": "validate_pdf",
                "failed": "retry_latex"
            }
        )
        
        workflow.add_conditional_edges(
            "validate_pdf",
            self._route_after_pdf_validation,
            {
                "success": END,
                "retry": "retry_latex",
                "failed": "retry_latex"
            }
        )
        
        return workflow.compile()
    
    # ========== NODES ==========
    
    def _enhance_cv_node(self, state: CVGenerationState) -> CVGenerationState:
     """Node: Enhancement du CV - AVEC LOGS DÉTAILLÉS"""
     print(f"\n🔄 [ENHANCE] Tentative {state['retry_count'] + 1}/{state['max_retries']}")
    
     try:
         enhanced_cv = self.agent.enhancer.enhance_cv_ethically(
             state['user_profile'],
             state['job_profile']
         )
        
        # 🔥 LOGS DÉBOGAGE DÉTAILLÉS
         print(f"📊 [ENHANCE DEBUG] Type du CV amélioré: {type(enhanced_cv)}")
         if isinstance(enhanced_cv, dict):
             print(f"📊 [ENHANCE DEBUG] Champs présents: {list(enhanced_cv.keys())}")
             print(f"📊 [ENHANCE DEBUG] Compétences techniques: {len(enhanced_cv.get('competences_techniques', []))} items")
             print(f"📊 [ENHANCE DEBUG] Expériences: {len(enhanced_cv.get('experiences', []))} items")
             print(f"📊 [ENHANCE DEBUG] Formation: {enhanced_cv.get('formation', 'N/A')}")
         else:
             print(f"❌ [ENHANCE DEBUG] enhanced_cv n'est pas un dict: {enhanced_cv}")
        
         state['enhanced_cv'] = enhanced_cv
         state['status'] = "pending"
         print("✅ [ENHANCE] CV amélioré généré")
     except Exception as e:
         print(f"❌ [ENHANCE] Erreur: {e}")
         state['status'] = "failed"
         state['validation_errors'].append(f"Enhancement error: {str(e)}")
    
     return state
    
    def _validate_json_node(self, state: CVGenerationState) -> CVGenerationState:
     """Node: Validation du JSON - VERSION SIMPLE"""
     print("\n🔍 [VALIDATE JSON] Vérification simple...")
    
     enhanced_cv = state['enhanced_cv']
    
    # 🔥 AFFICHER LA RÉPONSE COMPLÈTE DU LLM
     print("📋 [LLM RESPONSE] CV amélioré reçu:")
     print(json.dumps(enhanced_cv, indent=2, ensure_ascii=False)[:1000] + "..." if len(json.dumps(enhanced_cv)) > 1000 else "")
    
     errors = []
    
    # Vérification 1: Est-ce un dict?
     if not isinstance(enhanced_cv, dict):
         errors.append("❌ N'est pas un dictionnaire")
         print("❌ ERREUR: enhanced_cv n'est pas un dict")
    
    # Vérification 2: Champs obligatoires existent et non vides
     required_fields = ['competences_techniques', 'experiences']
    
     for field in required_fields:
         if field not in enhanced_cv:
             errors.append(f"❌ Champ '{field}' manquant")
             print(f"❌ Champ manquant: {field}")
         elif not enhanced_cv[field]:
             errors.append(f"❌ Champ '{field}' vide")
             print(f"❌ Champ vide: {field}")
         else:
             print(f"✅ Champ OK: {field}")
    
    # Vérification 3: Les listes sont des listes
     if isinstance(enhanced_cv.get('competences_techniques'), list):
         print(f"✅ competences_techniques: {len(enhanced_cv['competences_techniques'])} compétences")
     else:
         errors.append("❌ competences_techniques n'est pas une liste")
         print("❌ competences_techniques n'est pas une liste")
    
     if isinstance(enhanced_cv.get('experiences'), list):
         print(f"✅ experiences: {len(enhanced_cv['experiences'])} expériences")
     else:
         errors.append("❌ experiences n'est pas une liste")
         print("❌ experiences n'est pas une liste")
    
     state['validation_errors'] = errors
    
    # DÉCISION SIMPLE: Si pas d'erreurs critiques → VALIDE
     if not errors:
         print("🎯 [VALIDATE JSON] JSON VALIDE - Continuer vers LaTeX")
         state['status'] = "pending"
     else:
         print(f"🚫 [VALIDATE JSON] JSON INVALIDE - {len(errors)} erreurs")
         state['status'] = "json_invalid"
    
     return state
    
    def _generate_latex_node(self, state: CVGenerationState) -> CVGenerationState:
        """Node: Génération LaTeX"""
        print("\n📝 [LATEX] Génération...")
        
        try:
            # Ajouter instruction de compression si retry
            compress_mode = state['retry_count'] > 0
            
            latex_content = self.agent.latex_generator.generate_latex_cv(
                state['enhanced_cv'],
                state.get('job_profile')
            )
            
            # Si retry, ajouter instruction de compression dans le template
            if compress_mode:
                print("🗜️ [LATEX] Mode compression activé")
                # Vous pouvez ajuster le template ici si besoin
            
            state['latex_content'] = latex_content
            state['status'] = "pending"
            print("✅ [LATEX] Contenu généré")
        except Exception as e:
            print(f"❌ [LATEX] Erreur: {e}")
            state['status'] = "failed"
            state['validation_errors'].append(f"LaTeX generation error: {str(e)}")
        
        return state
    
    def _validate_latex_node(self, state: CVGenerationState) -> CVGenerationState:
        """Node: Validation du LaTeX"""
        print("\n🔍 [VALIDATE LATEX] Vérification...")
        
        latex_content = state['latex_content']
        errors = []
        
        # Vérifier que les sections principales sont présentes et non vides
        required_sections = [
            (r'\section{Expérience Professionnelle}', 'Expérience'),
            (r'\section{Work Experience}', 'Work Experience'),
            (r'\section{Compétences}', 'Compétences'),
            (r'\section{Skills}', 'Skills')
        ]
        
        has_experience = False
        has_skills = False
        
        for pattern, name in required_sections:
            if pattern in latex_content:
                # Vérifier que la section n'est pas vide
                section_start = latex_content.find(pattern)
                next_section = latex_content.find(r'\section{', section_start + 1)
                
                if next_section == -1:
                    section_content = latex_content[section_start:]
                else:
                    section_content = latex_content[section_start:next_section]
                
                # Compter le contenu réel (pas juste les commandes LaTeX)
                content_lines = [line for line in section_content.split('\n') 
                                if line.strip() and not line.strip().startswith('\\')]
                
                if len(content_lines) < 3:  # Au moins 3 lignes de contenu réel
                    errors.append(f"Section '{name}' quasi-vide")
                else:
                    if 'Experience' in name or 'Expérience' in name:
                        has_experience = True
                    if 'Skills' in name or 'Compétences' in name:
                        has_skills = True
        
        if not has_experience:
            errors.append("Section expérience manquante ou vide")
        
        if not has_skills:
            errors.append("Section compétences manquante ou vide")
        
        state['validation_errors'] = errors
        
        if errors:
            print(f"⚠️ [VALIDATE LATEX] {len(errors)} erreurs trouvées:")
            for err in errors:
                print(f"   - {err}")
            state['status'] = "pending"
        else:
            print("✅ [VALIDATE LATEX] LaTeX valide")
            state['status'] = "pending"
        
        return state
    
    def _compile_pdf_node(self, state: CVGenerationState) -> CVGenerationState:
        """Node: Compilation PDF"""
        print("\n🔧 [COMPILE] Compilation du PDF...")
        
        try:
            # Sauvegarder le fichier .tex
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tex_filename = f"enhanced_cv_{timestamp}.tex"
            
            self.agent.latex_generator.save_latex_to_file(
                state['latex_content'],
                tex_filename
            )
            
            # Compiler
            pdf_path = self.agent.latex_generator.compile_to_pdf(tex_filename)
            
            if pdf_path:
                state['pdf_path'] = pdf_path
                state['status'] = "pending"
                print(f"✅ [COMPILE] PDF généré: {pdf_path}")
            else:
                state['status'] = "pending"
                state['validation_errors'].append("Compilation PDF échouée")
                print("❌ [COMPILE] Échec de compilation")
        except Exception as e:
            print(f"❌ [COMPILE] Erreur: {e}")
            state['status'] = "pending"
            state['validation_errors'].append(f"Compilation error: {str(e)}")
        
        return state
    
    def _validate_pdf_node(self, state: CVGenerationState) -> CVGenerationState:
     """Node: Validation du PDF - VERSION SIMPLE"""
     print("\n🔍 [VALIDATE PDF] Vérification simple...")
    
     pdf_path = state.get('pdf_path')
    
    # Vérification 1: Fichier existe
     if not pdf_path or not os.path.exists(pdf_path):
         print("❌ PDF introuvable")
         state['status'] = "failed"
         state['validation_errors'].append("PDF introuvable")
         return state
    
     print(f"✅ PDF trouvé: {pdf_path}")
    
     try:
        # Vérification 2: Nombre de pages = 1
         with open(pdf_path, 'rb') as f:
             pdf_reader = PyPDF2.PdfReader(f)
             num_pages = len(pdf_reader.pages)
        
         print(f"📄 Nombre de pages: {num_pages}")
        
         if num_pages == 1:
             print("🎯 [VALIDATE PDF] PDF VALIDE - 1 page")
             state['status'] = "success"
         else:
             print(f"⚠️ PDF a {num_pages} pages au lieu d'1")
             state['status'] = "pdf_multipage"
             state['validation_errors'].append(f"PDF contient {num_pages} pages")
            
     except Exception as e:
         print(f"❌ Erreur lecture PDF: {e}")
         state['status'] = "failed"
         state['validation_errors'].append(f"Erreur PDF: {str(e)}")
    
     return state
    
    def _retry_enhancement_node(self, state: CVGenerationState) -> CVGenerationState:
        """Node: Préparer retry enhancement"""
        state['retry_count'] += 1
        print(f"\n🔄 [RETRY] Préparation retry enhancement ({state['retry_count']}/{state['max_retries']})")
        state['validation_errors'] = []
        return state
    
    def _retry_latex_node(self, state: CVGenerationState) -> CVGenerationState:
        """Node: Préparer retry LaTeX avec compression"""
        state['retry_count'] += 1
        print(f"\n🔄 [RETRY] Préparation retry LaTeX avec compression ({state['retry_count']}/{state['max_retries']})")
        state['validation_errors'] = []
        return state
    
    # ========== ROUTING FUNCTIONS ==========
    
    def _route_after_enhancement(self, state: CVGenerationState) -> str:
        if state['status'] == "failed":
            return "failed"
        return "validate"
    
    def _route_after_json_validation(self, state: CVGenerationState) -> str:
     """Routing après validation JSON - SIMPLE"""
     if state['status'] == "json_invalid":
         if state['retry_count'] < state['max_retries']:
             print("🔄 Retry après JSON invalide")
             return "retry"
         else:
             print("❌ Échec final après retry JSON")
             return "failed"
     return "valid"
    
    def _route_after_latex_generation(self, state: CVGenerationState) -> str:
        if state['status'] == "failed":
            return "failed"
        return "validate"
    
    def _route_after_latex_validation(self, state: CVGenerationState) -> str:
        if state['status'] == "latex_invalid":
            if state['retry_count'] < state['max_retries']:
                return "retry"
            return "failed"
        return "valid"
    
    def _route_after_compilation(self, state: CVGenerationState) -> str:
        if state['status'] == "failed":
            return "failed"
        return "validate"
    
    def _route_after_pdf_validation(self, state: CVGenerationState) -> str:
     """Routing après validation PDF - SIMPLE"""
     if state['status'] == "success":
         return "success"
     elif state['status'] == "pdf_multipage":
         if state['retry_count'] < state['max_retries']:
             print("🔄 Retry après PDF multipage")
             return "retry"
         else:
             print("❌ Échec final après retry PDF")
             return "failed"
     return "failed"
    
    # ========== PUBLIC METHOD ==========
    
    def run_orchestrated_generation(self, user_profile: dict, job_profile: dict, max_retries: int = 2):
        """
        Lance la génération orchestrée avec validation et retry
        
        Args:
            user_profile: Profil utilisateur analysé
            job_profile: Profil du job analysé
            max_retries: Nombre maximum de tentatives (défaut: 2)
        
        Returns:
            dict: Résultat final avec statut et fichiers générés
        """
        print("\n" + "="*70)
        print("🎯 ORCHESTRATION LANGGRAPH - Génération CV avec validation")
        print("="*70)

        sys.stdout.flush()
        
        # État initial
        initial_state: CVGenerationState = {
            "user_profile": user_profile,
            "job_profile": job_profile,
            "enhanced_cv": {},
            "latex_content": "",
            "pdf_path": "",
            "validation_errors": [],
            "retry_count": 0,
            "max_retries": max_retries,
            "status": "pending"
        }
        
        sys.stdout.flush()
        # Exécuter le graph
        final_state = self.graph.invoke(initial_state)

        sys.stdout.flush()
        
        # Préparer le résultat
        print("\n" + "="*70)
        if final_state['status'] == "success":
            print("✅ SUCCÈS - CV généré et validé")
            return {
                "success": True,
                "enhanced_cv": final_state['enhanced_cv'],
                "pdf_path": final_state['pdf_path'],
                "total_retries": final_state['retry_count'],
                "status": "success"
            }
        else:
            print(f"❌ ÉCHEC - Statut: {final_state['status']}")
            print(f"   Tentatives effectuées: {final_state['retry_count']}/{max_retries}")
            if final_state['validation_errors']:
                print("   Erreurs:")
                for err in final_state['validation_errors']:
                    print(f"   - {err}")
            
            return {
                "success": False,
                "status": final_state['status'],
                "errors": final_state['validation_errors'],
                "total_retries": final_state['retry_count'],
                "partial_cv": final_state.get('enhanced_cv'),
                "partial_pdf": final_state.get('pdf_path')
            }











# ================================
# ================================
# ================================
# ================================

# Routes Flask
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/api/process-cv', methods=['POST'])
def process_cv():
    try:
        if 'cv_file' not in request.files:
            return jsonify({'error': 'Aucun fichier uploadé'}), 400
        
        file = request.files['cv_file']
        if file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Seuls les fichiers PDF sont autorisés'}), 400
        
        # Sauvegarder le fichier
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
        file.save(file_path)
        
        # Initialiser l'agent
        config = CareerAgentConfig()
        agent = EnhancedCareerAgent(config)
        
        # Traitement avec capture de la sortie
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            result = agent.process_pdf_cv_orchestrated(file_path, {})
        
        # Nettoyer le fichier uploadé
        try:
            os.remove(file_path)
        except:
            pass
        
        if result.get("success"):
            # Préparer les données pour l'affichage
            enhanced_cv = result['enhanced_cv_json']
            job_info = result['recommended_job']
            
            
            response_data = {
                'success': True,
                'candidate_info': {
                    'nom': enhanced_cv.get('nom_complet', 'Non spécifié'),
                    'email': enhanced_cv.get('email', 'Non spécifié'),
                    'telephone': enhanced_cv.get('telephone', 'Non spécifié'),
                    'poste': enhanced_cv.get('job_title', 'Non spécifié'),
                    'localisation': enhanced_cv.get('localisation', 'Non spécifié')
                },
                'job_info': {
                    'titre': job_info.get('title', 'Non spécifié'),
                    'entreprise': job_info.get('company', 'Non spécifié'),
                    'localisation': job_info.get('location', 'Non spécifié'),
                    'lien': job_info.get('url', '#')
                },
                'scores': {
                    'matching': round(result.get('match_score', 0) * 100, 1),
                    'contextuel': round(result.get('contextual_score', 0) * 100, 1)
                },
                'competences_ameliorees': enhanced_cv.get('competences_techniques', [])[:10],
                'experiences_ameliorees': enhanced_cv.get('experiences', [])[:5],
                'formation_amelioree': enhanced_cv.get('formation', ''),
                'objectif_ameliore': enhanced_cv.get('objectif_professionnel', ''),
                'fichiers_generes': {
                    'latex': result.get('latex_filename'),
                    'pdf': result.get('pdf_filename')
                },
                'logs': output.getvalue()[-2000:]  # Derniers 2000 caractères
            }
            
            return jsonify(response_data)
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Erreur inconnue lors du traitement')
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Erreur lors du traitement: {str(e)}'
        }), 500

@app.route('/api/download/<path:filename>')
def download_file(filename):
    try:
        # Sécuriser le chemin du fichier
        safe_path = os.path.join(os.getcwd(), filename)
        if os.path.exists(safe_path) and os.path.isfile(safe_path):
            return send_file(safe_path, as_attachment=True)
        else:
            return jsonify({'error': 'Fichier non trouvé'}), 404
    except Exception as e:
        return jsonify({'error': f'Erreur de téléchargement: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'Fichier trop volumineux. Maximum 16MB.'}), 413


# ================================
# NOUVELLE ROUTE POUR LA RÉGÉNÉRATION
# ================================

@app.route('/api/regenerate-enhancement', methods=['POST'])
def regenerate_enhancement():
    try:
        data = request.json
        cv_data = data.get('cv_data')
        job_data = data.get('job_data')
        
        if not cv_data or not job_data:
            return jsonify({'error': 'Données manquantes'}), 400
        
        # Initialiser l'agent
        config = CareerAgentConfig()
        agent = EnhancedCareerAgent(config)
        
        # Analyser le job
        job_analysis = agent.job_analyzer.analyze_job(job_data)
        
        if not job_analysis:
            return jsonify({'error': 'Erreur analyse job'}), 500
        
        # Régénérer l'enhancement
        enhanced_cv = agent.enhancer.enhance_cv_ethically(cv_data, job_analysis)
        
        if not enhanced_cv:
            return jsonify({'error': 'Erreur enhancement'}), 500
        
        # Générer le LaTeX et PDF
        latex_content = agent.latex_generator.generate_latex_cv(enhanced_cv, job_analysis)
        
        if not latex_content:
            return jsonify({'error': 'Erreur génération LaTeX'}), 500
        
        # Sauvegarder les fichiers
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tex_filename = f"regenerated_cv_{timestamp}.tex"
        agent.latex_generator.save_latex_to_file(latex_content, tex_filename)
        
        # Compiler en PDF
        pdf_filename = agent.latex_generator.compile_to_pdf(tex_filename)
        
        return jsonify({
            'success': True,
            'enhanced_cv': enhanced_cv,
            'fichiers_generes': {
                'latex': tex_filename,
                'pdf': pdf_filename
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Erreur lors de la régénération: {str(e)}'
        }), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)