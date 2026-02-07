from flask import Flask, render_template, request, send_file
import os
import csv
import PyPDF2
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ---------------- CONFIG ---------------- #
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

LATEST_RESULTS = []

# ---------------- ROUTES ---------------- #

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    global LATEST_RESULTS

    if request.method == "POST":
        resumes = request.files.getlist("resumes")
        job_desc = request.form.get("job_desc", "").strip()

        if not resumes or job_desc == "":
            return "Missing resumes or job description"

        jd_keywords = extract_keywords(job_desc)
        results = []

        for resume in resumes:
            if resume.filename == "":
                continue

            path = os.path.join(app.config["UPLOAD_FOLDER"], resume.filename)
            resume.save(path)

            resume_text = extract_text_from_pdf(path)

            # ---------- SCORES ---------- #
            skill_score = skill_match_score(resume_text)
            semantic_score = calculate_similarity(resume_text, job_desc) * 100
            strength_score = resume_strength_score(resume_text)

            # ---------- FINAL ATS SCORE ---------- #
            final_score = round(
                (skill_score * 0.5) +
                (semantic_score * 0.3) +
                (strength_score * 0.2),
                2
            )

            # ---------- DECISION ---------- #
            if final_score >= 65:
                status = "SELECTED"
                decision = "Eligible for Job"
            elif final_score >= 50:
                status = "BORDERLINE"
                decision = "Needs HR Review"
            else:
                status = "NOT SELECTED"
                decision = "Not Eligible"

            results.append({
                "filename": resume.filename,
                "score": final_score,
                "status": status,
                "decision": decision
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        LATEST_RESULTS = results

        return render_template("result.html", results=results)

    return render_template("upload.html")


@app.route("/download_csv")
def download_csv():
    if not LATEST_RESULTS:
        return "No results available"

    with open("ATS_Results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Resume", "ATS Score", "Status", "Decision"])

        for i, r in enumerate(LATEST_RESULTS, start=1):
            writer.writerow([i, r["filename"], r["score"], r["status"], r["decision"]])

    return send_file("ATS_Results.csv", as_attachment=True)


# ---------------- FUNCTIONS ---------------- #

def extract_text_from_pdf(path):
    text = ""
    try:
        with open(path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + " "
    except:
        pass
    return text.lower()


def extract_keywords(text):
    return set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))


def skill_match_score(resume_text):
    skills = [
        "python", "java", "javascript", "react", "node", "express",
        "mongodb", "sql", "machine learning", "deep learning",
        "tensorflow", "flask", "api", "html", "css", "mern",
        "ai", "nlp", "computer vision", "internship", "project"
    ]

    matched = sum(1 for skill in skills if skill in resume_text)
    return min((matched / len(skills)) * 100, 100)


def resume_strength_score(resume_text):
    score = 0

    if "project" in resume_text:
        score += 25
    if "intern" in resume_text or "internship" in resume_text:
        score += 25
    if "hackathon" in resume_text:
        score += 20
    if "certification" in resume_text or "coursera" in resume_text:
        score += 15
    if "github" in resume_text:
        score += 15

    return min(score, 100)


def calculate_similarity(resume_text, job_desc):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume_text, job_desc])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    return similarity[0][0]


# ---------------- MAIN ---------------- #

if __name__ == "__main__":
    app.run(debug=True)








