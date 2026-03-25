from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
import re

app = FastAPI()
print("App is starting...")

model = None

def get_model():
    global model
    if model is None:
        print("Loading model...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
    return model
    
skills_list = [
    "python", "java", "machine learning", "deep learning",
    "data analysis", "sql", "tensorflow", "pytorch", "computer vision"
]

# ---- helper functions ----

def split_sentences(text):
    sentences = re.split(r'[.,\n]+', text.lower())
    return [s.strip() for s in sentences if s.strip()]


def calculate_similarity(resume, job):
    embeddings = get_model().encode([resume, job])
    return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0] * 100)


def extract_skills(text, skills_list):
    text = text.lower()
    return [skill for skill in skills_list if skill in text]


def extract_skills_hybrid(text, skills_list, threshold=0.4):
    text_lower = text.lower()
    exact = [s for s in skills_list if s in text_lower]

    sentences = split_sentences(text)
    if not sentences:
        return exact

    sent_emb = get_model().encode(sentences)
    semantic = []

    for skill in skills_list:
        if skill in exact:
            continue

        skill_emb = get_model().encode([skill])[0]

        for emb in sent_emb:
            sim = cosine_similarity([emb], [skill_emb])[0][0]
            if sim > threshold:
                semantic.append(skill)
                break

    return list(set(exact + semantic))


def skill_match_score(resume_skills, job_skills):
    if not job_skills:
        return 0
    return (len(set(resume_skills) & set(job_skills)) / len(job_skills)) * 100


def final_score(resume, job):
    sim = calculate_similarity(resume, job)

    res_skills = extract_skills_hybrid(resume, skills_list)
    jd_skills = extract_skills(job, skills_list)

    skill_score = skill_match_score(res_skills, jd_skills)

    final = 0.6 * sim + 0.4 * skill_score

    return {
        "semantic_score": float(round(sim, 2)),
        "skill_score": float(round(skill_score, 2)),
        "final_score": float(round(final, 2)),
        "missing_skills": list(set(jd_skills) - set(res_skills))
    }


# ---- API ----

class RequestData(BaseModel):
    resume: str
    job_description: str


@app.get("/")
def home():
    return {"message": "API is working"}
    
@app.post("/analyze")
def analyze(data: RequestData):
    return {"status": "working"}
    
# @app.post("/analyze")
# def analyze(data: RequestData):
#     return final_score(data.resume, data.job_description)
