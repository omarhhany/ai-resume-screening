# AI Resume Screening System

An AI-powered web application that ranks resumes based on job descriptions using Natural Language Processing (NLP).

## 🚀 Live Demo
👉 https://ai-resume-screening-8wjv4f9wpp5sh4tzblnqpk.streamlit.app/

## 🧠 Features
- Upload multiple PDF resumes
- Automatic text extraction from PDFs
- Resume-job matching using TF-IDF
- Skill-based scoring system
- Candidate ranking based on relevance
- Interactive web interface built with Streamlit

## 🛠️ Tech Stack
- Python
- Scikit-learn
- NLTK
- Streamlit
- PDFMiner

## ⚙️ How It Works
1. Extract text from uploaded PDF resumes  
2. Clean and preprocess text  
3. Convert text into TF-IDF vectors  
4. Compare with job description using cosine similarity  
5. Extract and match relevant skills  
6. Rank candidates based on combined score  

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py