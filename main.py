from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Skills Database ----
skills_list = [
    "python", "java", "machine learning", "deep learning",
    "data analysis", "sql", "tensorflow", "pytorch", "computer vision"
]

# ---- Helper Functions ----

def split_sentences(text):
    sentences = re.split(r'[.,\n]+', text.lower())
    return [s.strip() for s in sentences if s.strip()]


import requests

API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": "Bearer YOUR_API_KEY"}

def get_embedding(text):
    response = requests.post(API_URL, headers=headers, json={"inputs": text})
    return response.json()

def calculate_similarity(resume, job):
    emb1 = get_embedding(resume)
    emb2 = get_embedding(job)

    sim = cosine_similarity([emb1], [emb2])[0][0]
    return float(sim * 100)


def extract_skills(text, skills_list):
    text = text.lower()
    return [skill for skill in skills_list if skill in text]


def extract_skills_hybrid(text, skills_list):
    # For TF-IDF version, keep it simple (no embeddings)
    return extract_skills(text, skills_list)


def skill_match_score(resume_skills, job_skills):
    if not job_skills:
        return 0
    return (len(set(resume_skills) & set(job_skills)) / len(job_skills)) * 100

def generate_suggestions(missing):
    return [f"Consider adding {skill} to your resume" for skill in missing]


def final_score(resume, job):
    sim = calculate_similarity(resume, job)

    res_skills = extract_skills_hybrid(resume, skills_list)
    jd_skills = extract_skills(job, skills_list)

    skill_score = skill_match_score(res_skills, jd_skills)

    final = 0.6 * sim + 0.4 * skill_score
    missing_skills = list(set(jd_skills) - set(res_skills))
    return {
        "semantic_score": float(round(sim, 2)),
        "skill_score": float(round(skill_score, 2)),
        "final_score": float(round(final, 2)),
        "matched_skills": list(set(res_skills) & set(jd_skills)),
        "missing_skills": missing_skills,
        "suggestions": generate_suggestions(missing_skills)
    }

# ---- API Schema ----

class RequestData(BaseModel):
    resume: str
    job_description: str


# ---- Routes ----

@app.get("/")
def home():
    return {"message": "AI Resume Screener API is running 🚀"}


@app.post("/analyze")
def analyze(data: RequestData):
    return final_score(data.resume, data.job_description)
