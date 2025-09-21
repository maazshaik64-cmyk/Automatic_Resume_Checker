import streamlit as st
from pathlib import Path
import re
import fitz                # PyMuPDF
import pdfplumber
from docx import Document
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import sqlite3
import json
from datetime import datetime

# ----------------------------
# DATABASE
# ----------------------------
conn = sqlite3.connect("evaluations.db", check_same_thread=False)
conn.execute("""
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    jd_file TEXT,
    resume_file TEXT,
    job_title TEXT,
    resume_content TEXT,
    jd_content TEXT,
    hard_score REAL,
    semantic_score REAL,
    final_score REAL,
    verdict TEXT,
    missing_skills TEXT,
    suggestions TEXT
)
""")
conn.commit()

# ----------------------------
# 1. TEXT EXTRACTION
# ----------------------------
def extract_text_from_pdf(path):
    try:
        doc = fitz.open(path)
        text = [page.get_text("text") for page in doc]
        return "\n".join(text).strip()
    except Exception:
        try:
            with pdfplumber.open(path) as pdf:
                text = [p.extract_text() or "" for p in pdf.pages]
            return "\n".join(text).strip()
        except Exception:
            return ""

def extract_text_from_docx(path):
    try:
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs]).strip()
    except Exception:
        return ""

# ----------------------------
# 2. NORMALIZATION
# ----------------------------
def normalize_text(raw):
    txt = raw.replace('\r', '\n')
    txt = re.sub(r'\n{2,}', '\n\n', txt)
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    lines = [l for l in lines if not re.match(r'^(page|pg)\b', l.lower())]
    return "\n".join(lines)

# ----------------------------
# 3. JD PARSING (simple skills extractor)
# ----------------------------
def extract_skills_from_jd(jd_text):
    skills = set()
    lines = jd_text.splitlines()
    for l in lines:
        if re.search(r'\b(skill|requirement|qualification|responsibil|must have|good to have)\b', l, re.I):
            parts = re.split(r'[•\-\n,;]', l)
            for p in parts:
                p = p.strip()
                if 2 <= len(p) <= 60:
                    if not re.match(r'^(and|or|the|with|a|an|for)$', p, re.I):
                        skills.add(p)
    return list(skills)

# ----------------------------
# 4. HARD MATCHING
# ----------------------------
def fuzzy_match_skill(skill, resume_text, threshold=75):
    if re.search(r'\b' + re.escape(skill) + r'\b', resume_text, re.I):
        return 1.0
    score = fuzz.partial_ratio(skill.lower(), resume_text.lower())
    return score / 100.0 if score >= threshold else 0.0

def compute_hard_score(jd_must, jd_good, resume_text):
    must_scores = [fuzzy_match_skill(s, resume_text) for s in jd_must]
    good_scores = [fuzzy_match_skill(s, resume_text) for s in jd_good]
    must_sum = sum(must_scores)
    good_sum = sum(good_scores)
    must_total = len(jd_must) if jd_must else 1
    good_total = len(jd_good) if jd_good else 1
    must_part = (must_sum / must_total) * 70.0
    good_part = (good_sum / good_total) * 30.0
    return must_part + good_part, must_scores, good_scores

# ----------------------------
# 5. SEMANTIC MATCHING
# ----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

def embed_text(text):
    return model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

def semantic_score(jd_text, resume_text):
    v_jd = embed_text(jd_text)
    v_res = embed_text(resume_text)
    sim = float(np.dot(v_jd, v_res))
    mapped = (max(min(sim, 1.0), -1.0) + 1.0) / 2.0 * 100.0
    return mapped

# ----------------------------
# 6. EVALUATION PIPELINE
# ----------------------------
def evaluate_resume_against_jd(resume_text, jd_text, jd_must, jd_good, w_hard=0.6, w_sem=0.4):
    hard_score, must_scores, good_scores = compute_hard_score(jd_must, jd_good, resume_text)
    sem_score = semantic_score(jd_text, resume_text)
    final = w_hard * hard_score + w_sem * sem_score
    missing = [jd_must[i] for i,s in enumerate(must_scores) if s < 0.5]
    suggestions = []
    if sem_score < 60:
        suggestions.append("Improve summary and add role-specific keywords/projects.")
    if missing:
        suggestions.append("Add or highlight these must-have skills: " + ", ".join(missing))
    if hard_score < 50:
        suggestions.append("Add concrete metrics/descriptions for relevant skills/projects.")
    if final >= 75:
        verdict = "High"
    elif final >= 50:
        verdict = "Medium"
    else:
        verdict = "Low"
    return {
        "hard_score": round(hard_score,2),
        "semantic_score": round(sem_score,2),
        "final_score": round(final,2),
        "verdict": verdict,
        "missing_skills": missing,
        "suggestions": suggestions
    }

# ----------------------------
# 7. LLM FEEDBACK FUNCTION (LOCAL FLAN-T5)
# ----------------------------
@st.cache_resource
def load_feedback_model():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, feedback_model = load_feedback_model()

def generate_feedback(jd_text, resume_text, missing_skills, score, verdict):
    prompt = f"""
You are an expert career coach.
Job Description:
{jd_text}

Resume:
{resume_text}

The resume scored {score} with verdict: {verdict}.
Missing skills: {missing_skills}.

Provide 3-4 bullet points of constructive feedback to improve the resume.
Start each point with a hyphen (-).
"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = feedback_model.generate(**inputs, max_new_tokens=200)
        feedback = tokenizer.decode(outputs[0], skip_special_tokens=True)
        suggestions = [s.strip() for s in feedback.split('-') if s.strip()]
        return suggestions
    except Exception as e:
        st.error(f"LLM feedback generation failed: {e}")
        return ["Could not generate feedback"]

# ----------------------------
# 8. STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="Automated Resume Checker", layout="wide")
st.title("🤖 Automated Resume Checker")

menu = st.sidebar.radio("Menu", ["Evaluate Resume", "Dashboard"])

# ----------------------------
# EVALUATE RESUME VIEW
# ----------------------------
if menu == "Evaluate Resume":
    uploaded_jd = st.file_uploader("📄 Upload Job Description (txt/pdf/docx)", type=["pdf","docx","txt"])
    uploaded_resume = st.file_uploader("🧑‍🎓 Upload Resume (txt/pdf/docx)", type=["pdf","docx","txt"])

    if uploaded_jd and uploaded_resume:
        jd_path = Path("tmp_jd." + uploaded_jd.name.split('.')[-1])
        resume_path = Path("tmp_resume." + uploaded_resume.name.split('.')[-1])
        jd_path.write_bytes(uploaded_jd.getvalue())
        resume_path.write_bytes(uploaded_resume.getvalue())

        try:
            if jd_path.suffix == ".pdf":
                jd_raw = extract_text_from_pdf(str(jd_path))
            elif jd_path.suffix == ".docx":
                jd_raw = extract_text_from_docx(str(jd_path))
            else:
                jd_raw = jd_path.read_text(encoding="utf-8")
        except:
            st.error("Failed to read JD file.")
            jd_raw = ""

        try:
            if resume_path.suffix == ".pdf":
                resume_raw = extract_text_from_pdf(str(resume_path))
            elif resume_path.suffix == ".docx":
                resume_raw = extract_text_from_docx(str(resume_path))
            else:
                resume_raw = resume_path.read_text(encoding="utf-8")
        except:
            st.error("Failed to read Resume file.")
            resume_raw = ""

        jd_norm = normalize_text(jd_raw)
        resume_norm = normalize_text(resume_raw)

        # Extract skills
        jd_skills = extract_skills_from_jd(jd_norm)
        jd_must = jd_skills[:5]
        jd_good = jd_skills[5:10]

        st.subheader("🔎 Extracted JD Skills")
        st.write("Must-have:", jd_must)
        st.write("Good-to-have:", jd_good)

        # Evaluate resume
        result = evaluate_resume_against_jd(resume_norm, jd_norm, jd_must, jd_good)

        st.subheader("📊 Results")
        if result["verdict"] == "High":
            st.success(f"Verdict: **{result['verdict']}**")
        elif result["verdict"] == "Medium":
            st.warning(f"Verdict: **{result['verdict']}**")
        else:
            st.error(f"Verdict: **{result['verdict']}**")

        st.write("Hard score:", result["hard_score"], "| Semantic score:", result["semantic_score"])
        st.write("Missing skills:", result["missing_skills"])
        st.write("Suggestions:")
        for s in result["suggestions"]:
            st.write("✅", s)

        # Generate LLM feedback
        st.subheader("🤖 LLM Feedback")
        feedback = generate_feedback(jd_norm, resume_norm, result["missing_skills"], result["final_score"], result["verdict"])
        for f in feedback:
            st.write("✅", f)

        # Extract job title
        job_title = jd_norm.splitlines()[0] if jd_norm else "Unknown"
        match = re.search(r'(Position|Role|Job Title)[:\-]\s*(.+)', jd_norm, re.I)
        if match:
            job_title = match.group(2).strip()

        # Save evaluation to DB
        conn.execute("""
            INSERT INTO evaluations (
                timestamp, jd_file, resume_file, job_title,
                resume_content, jd_content,
                hard_score, semantic_score, final_score, verdict,
                missing_skills, suggestions
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            uploaded_jd.name,
            uploaded_resume.name,
            job_title,
            resume_norm,
            jd_norm,
            result["hard_score"],
            result["semantic_score"],
            result["final_score"],
            result["verdict"],
            json.dumps(result["missing_skills"]),
            json.dumps(result["suggestions"])
        ))
        conn.commit()

# ----------------------------
# DASHBOARD VIEW
# ----------------------------
elif menu == "Dashboard":
    st.subheader("📋 Resume Evaluations Dashboard")
    # Filter by job title
    jobs = [row[0] for row in conn.execute("SELECT DISTINCT job_title FROM evaluations").fetchall()]
    selected_job = st.selectbox("Select Job Description", ["All"] + jobs)

    query = "SELECT id, timestamp, jd_file, resume_file, job_title, final_score, verdict FROM evaluations"
    if selected_job != "All":
        query += f" WHERE job_title = '{selected_job}'"

    rows = conn.execute(query).fetchall()
    import pandas as pd
    df = pd.DataFrame(rows, columns=["ID","Timestamp","JD File","Resume File","Job Title","Final Score","Verdict"])
    st.dataframe(df.sort_values(by="Final Score", ascending=False), use_container_width=True)







