from flask import Flask, render_template, request, redirect, url_for, session
from PyPDF2 import PdfReader
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import os
import re
import sqlite3

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

app.secret_key = "smarthire-secret-key"

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==================================================
# DATABASE
# ==================================================

def get_db():

    conn = sqlite3.connect("database.db")

    conn.row_factory = sqlite3.Row

    return conn


def create_database():

    conn = get_db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    conn.commit()

    conn.close()


create_database()


# ==================================================
# JOB DATA
# ==================================================

jobs = pd.read_csv("data/jobs.csv")


# ==================================================
# SKILLS
# ==================================================

SKILLS = [
    "python",
    "java",
    "c",
    "c++",
    "sql",
    "mysql",
    "mongodb",
    "django",
    "flask",
    "fastapi",
    "html",
    "css",
    "javascript",
    "react",
    "node.js",
    "git",
    "github",
    "docker",
    "pandas",
    "numpy",
    "scikit-learn",
    "machine learning",
    "deep learning",
    "tensorflow",
    "pytorch",
    "statistics",
    "excel",
    "power bi",
    "data analysis",
    "rest api"
]


# ==================================================
# HOME
# ==================================================

@app.route("/")
def home():

    if "user_id" not in session:
        return redirect(url_for("login"))

    return redirect(url_for("dashboard"))


# ==================================================
# REGISTER
# ==================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]

        hashed_password = generate_password_hash(password)

        conn = get_db()

        try:

            conn.execute(
                """
                INSERT INTO users
                (name, email, password)
                VALUES (?, ?, ?)
                """,
                (name, email, hashed_password)
            )

            conn.commit()

            conn.close()

            return redirect(url_for("login"))

        except sqlite3.IntegrityError:

            conn.close()

            return render_template(
                "register.html",
                error="Email already registered"
            )

    return render_template("register.html")


# ==================================================
# LOGIN
# ==================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        conn = get_db()

        user = conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (email,)
        ).fetchone()

        conn.close()

        if user and check_password_hash(
            user["password"],
            password
        ):

            session["user_id"] = user["id"]
            session["user_name"] = user["name"]

            return redirect(url_for("dashboard"))

        return render_template(
            "login.html",
            error="Invalid email or password"
        )

    return render_template("login.html")


# ==================================================
# LOGOUT
# ==================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("login"))


# ==================================================
# DASHBOARD
# ==================================================

@app.route("/dashboard")
def dashboard():

    if "user_id" not in session:
        return redirect(url_for("login"))

    return render_template(
        "dashboard.html",
        name=session["user_name"]
    )


# ==================================================
# PDF TEXT EXTRACTION
# ==================================================

def extract_text_from_pdf(file_path):

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ==================================================
# SKILL EXTRACTION
# ==================================================

def extract_skills(text):

    text = text.lower()

    found_skills = []

    for skill in SKILLS:

        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

        if re.search(pattern, text):

            found_skills.append(skill)

    return sorted(set(found_skills))


# ==================================================
# RESUME SCORE
# ==================================================

def calculate_resume_score(found_skills):

    if not found_skills:

        return 0

    score = (
        len(found_skills) /
        len(SKILLS)
    ) * 100

    return round(
        min(score, 100),
        2
    )


# ==================================================
# JOB RECOMMENDATION
# ==================================================

def recommend_jobs(resume_text):

    documents = [
        resume_text
    ] + jobs["description"].fillna("").tolist()

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(
        documents
    )

    similarity_scores = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:]
    )[0]

    jobs_copy = jobs.copy()

    jobs_copy["match_score"] = (
        similarity_scores * 100
    ).round(2)

    jobs_copy = jobs_copy.sort_values(
        by="match_score",
        ascending=False
    )

    return jobs_copy.head(5)


# ==================================================
# SKILL GAP
# ==================================================

def skill_gap(found_skills, job_skills):

    required_skills = [
        skill.strip().lower()
        for skill in job_skills.split(",")
    ]

    matched = [
        skill
        for skill in required_skills
        if skill in found_skills
    ]

    missing = [
        skill
        for skill in required_skills
        if skill not in found_skills
    ]

    return matched, missing


# ==================================================
# RESUME ANALYSIS
# ==================================================

@app.route("/analyze", methods=["POST"])
def analyze():

    if "user_id" not in session:

        return redirect(url_for("login"))

    if "resume" not in request.files:

        return "No resume uploaded"

    file = request.files["resume"]

    if file.filename == "":

        return "Please select a resume"

    if not file.filename.lower().endswith(".pdf"):

        return "Only PDF files are allowed"

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(file_path)

    resume_text = extract_text_from_pdf(
        file_path
    )

    found_skills = extract_skills(
        resume_text
    )

    resume_score = calculate_resume_score(
        found_skills
    )

    recommended_jobs = recommend_jobs(
        resume_text
    )

    recommendations = []

    for _, job in recommended_jobs.iterrows():

        matched, missing = skill_gap(
            found_skills,
            job["skills"]
        )

        recommendations.append({

            "title": job["title"],

            "company": job["company"],

            "score": job["match_score"],

            "skills": job["skills"],

            "matched": matched,

            "missing": missing

        })

    return render_template(
        "result.html",

        resume_score=resume_score,

        skills=found_skills,

        recommendations=recommendations
    )


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":

    app.run(debug=True)