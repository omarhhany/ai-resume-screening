import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# DOWNLOAD STOPWORDS
# -------------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -------------------------------
# STEP 1: LOAD DATA
# -------------------------------
df = pd.read_csv("data/raw/Resume.csv", encoding='latin-1')

print("Columns:", df.columns)

# -------------------------------
# STEP 2: FILTER RELEVANT CATEGORIES
# -------------------------------
relevant_categories = ["INFORMATION-TECHNOLOGY", "ENGINEERING"]
df = df[df['Category'].isin(relevant_categories)]

# -------------------------------
# STEP 3: CLEAN TEXT
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df['cleaned'] = df['Resume_str'].apply(clean_text)

print("\nSample cleaned text:\n")
print(df['cleaned'].head())

# -------------------------------
# STEP 4: TF-IDF
# -------------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words='english'
)

X = vectorizer.fit_transform(df['cleaned'])

print("\nTF-IDF shape:", X.shape)

# -------------------------------
# STEP 5: JOB DESCRIPTION
# -------------------------------
job_description = """
Looking for a software engineer with strong experience in Python, machine learning,
data analysis, and deep learning. Candidate should be familiar with TensorFlow,
scikit-learn, and building predictive models.
"""

job_cleaned = clean_text(job_description)
job_vector = vectorizer.transform([job_cleaned])

# -------------------------------
# STEP 6: SIMILARITY
# -------------------------------
similarities = cosine_similarity(job_vector, X)

# -------------------------------
# STEP 7: SKILL EXTRACTION
# -------------------------------
skills_list = [
    "python", "java", "machine learning", "deep learning",
    "data analysis", "sql", "tensorflow", "pandas",
    "numpy", "scikit-learn"
]

def extract_skills(text):
    found = []
    for skill in skills_list:
        if skill in text:
            found.append(skill)
    return found

df['skills'] = df['cleaned'].apply(extract_skills)

# -------------------------------
# STEP 8: JOB SKILLS
# -------------------------------
job_skills = extract_skills(job_cleaned)
print("\nJob Skills:", job_skills)

# -------------------------------
# STEP 9: SKILL SCORE
# -------------------------------
def skill_score(resume_skills, job_skills):
    if len(job_skills) == 0:
        return 0
    match = len(set(resume_skills) & set(job_skills))
    return match / len(job_skills)

# -------------------------------
# STEP 10: FINAL RANKING
# -------------------------------
results = []

for i in range(len(df)):
    sim_score = similarities[0][i]
    skill_match = skill_score(df.iloc[i]['skills'], job_skills)
    
    final_score = 0.7 * sim_score + 0.3 * skill_match

    results.append((i, final_score))

# Sort results
results = sorted(results, key=lambda x: x[1], reverse=True)

# -------------------------------
# STEP 11: OUTPUT TOP MATCHES
# -------------------------------
print("\nTop Matches:\n")

for idx, score in results[:5]:
    print(f"Final Score: {score:.4f}")
    print("Category:", df.iloc[idx]['Category'])
    print("Skills:", df.iloc[idx]['skills'])
    print("-" * 40)