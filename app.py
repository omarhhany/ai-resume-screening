import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text

# -------------------------------
# SETUP
# -------------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -------------------------------
# CLEAN TEXT
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# -------------------------------
# SKILLS
# -------------------------------
skills_list = [
    "python", "java", "machine learning", "deep learning",
    "data analysis", "sql", "tensorflow", "pandas",
    "numpy", "scikit-learn"
]

def extract_skills(text):
    return [skill for skill in skills_list if skill in text]

def skill_score(resume_skills, job_skills):
    if len(job_skills) == 0:
        return 0
    return len(set(resume_skills) & set(job_skills)) / len(job_skills)

# -------------------------------
# PDF TEXT EXTRACTION
# -------------------------------
def extract_text_from_pdf(file):
    return extract_text(file)

# -------------------------------
# UI CONFIG
# -------------------------------
st.set_page_config(page_title="AI Resume Screening", layout="centered")

st.title("🤖 AI Resume Screening System")

st.markdown("Upload resumes and match them against a job description using AI.")

job_desc = st.text_area("📄 Enter Job Description")

uploaded_files = st.file_uploader(
    "📤 Upload PDF Resumes",
    type=["pdf"],
    accept_multiple_files=True
)

# -------------------------------
# MAIN LOGIC
# -------------------------------
if st.button("🚀 Analyze") and uploaded_files and job_desc:

    with st.spinner("Analyzing resumes..."):

        resumes = []
        filenames = []

        # Extract text
        for file in uploaded_files:
            try:
                text = extract_text_from_pdf(file)
                cleaned = clean_text(text)
                resumes.append(cleaned)
                filenames.append(file.name)
            except:
                st.warning(f"⚠️ Could not read {file.name}")

        if len(resumes) == 0:
            st.error("❌ No valid resumes processed.")
        else:
            # TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english'
            )

            X = vectorizer.fit_transform(resumes)

            # Job processing
            job_cleaned = clean_text(job_desc)
            job_vector = vectorizer.transform([job_cleaned])

            similarities = cosine_similarity(job_vector, X)

            # Skills
            resume_skills = [extract_skills(r) for r in resumes]
            job_skills = extract_skills(job_cleaned)

            # Ranking
            results = []

            for i in range(len(resumes)):
                sim_score = similarities[0][i]
                skill_match = skill_score(resume_skills[i], job_skills)
                final_score = 0.7 * sim_score + 0.3 * skill_match
                results.append((i, final_score))

            results = sorted(results, key=lambda x: x[1], reverse=True)

            # -------------------------------
            # OUTPUT
            # -------------------------------
            st.subheader("🏆 Top Candidates")

            # Warning if weak results
            if results[0][1] < 0.2:
                st.warning("⚠️ Low match scores — resumes may not strongly match the job description.")

            for idx, score in results[:3]:
                st.markdown(f"""
                ### 📄 {filenames[idx]}

                - **Match Score:** {score:.2f}
                - **Skills Found:** {', '.join(resume_skills[idx]) if resume_skills[idx] else 'None'}

                ---
                """)