import zipfile
import json
import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import streamlit as st

# Make sure NLTK downloads only once
nltk.download('punkt')
nltk.download('stopwords')

# Load stopwords
stop_words = set(stopwords.words('english'))

def load_jsons_from_zip(zip_filename):
    all_data = []

    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        for file_name in zipf.namelist():
            if file_name.endswith(".json"):
                with zipf.open(file_name) as f:
                    data = json.load(f)
                    all_data.append(data)

    df = pd.DataFrame(all_data)
    print(f"Loaded {len(df)} records from ZIP")
    return df

def clean_and_tokenize(text):
    if not isinstance(text, str):
        return []

    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in stop_words and len(word) > 2]
    return filtered

def score_resume(jd_tokens, resume_tokens, skill_terms=None):
    skill_terms = skill_terms or []
    jd_freq = Counter(jd_tokens)
    resume_freq = Counter(resume_tokens)
    matched_keywords = set(jd_tokens) & set(resume_tokens)
    raw_score = sum(resume_freq[word] * jd_freq[word] for word in matched_keywords)
    skill_bonus = sum(2 for skill in skill_terms if skill in resume_tokens)
    total_score = raw_score + skill_bonus
    max_score = sum(jd_freq.values()) + len(skill_terms) * 2
    percentage_score = (total_score / max_score) * 100 if max_score > 0 else 0
    return round(percentage_score, 2), list(matched_keywords)

def shortlist_resumes_by_jd(df, job_description, k=5):
    skill_terms = ["python", "sql", "api", "aws", "django", "flask", "excel", "sales", "marketing", "analysis"]
    jd_tokens = clean_and_tokenize(job_description)
    results = []

    for _, row in df.iterrows():
        resume_text = row['input'].get('resume', '')
        resume_tokens = clean_and_tokenize(resume_text)
        score, matched_keywords = score_resume(jd_tokens, resume_tokens, skill_terms)

        results.append({
            "name": row['details'].get('name', 'N/A'),
            "score": score,
            "matched_keywords": matched_keywords[:10],
            "resume_snippet": " ".join(resume_tokens[:50]) + "...",
            "details": row['details']
        })

    top_resumes = sorted(results, key=lambda x: x['score'], reverse=True)[:k]
    return top_resumes

if __name__ == "__main__":
    st.set_page_config(page_title="Resume Shortlisting System", layout="wide")
    st.title("📄 Resume Shortlisting System")

    uploaded_file = st.file_uploader("Upload ZIP file of resumes (.json inside)", type=["zip"])

    job_description = st.text_area(
        "Paste Job Description Below:",
        height=200,
        value="We are hiring a backend developer with experience in Python, Flask, API development, and database design.\nFamiliarity with AWS, Docker, and deployment pipelines is a plus."
    )

    k = st.slider("Number of Top Resumes to Display", 1, 20, 5)

    if st.button("Shortlist Resumes"):
        if uploaded_file is None:
            st.warning("Please upload a ZIP file of resumes.")
        elif not job_description.strip():
            st.warning("Please provide a job description.")
        else:
            try:
                df = load_jsons_from_zip(uploaded_file)
                top_resumes = shortlist_resumes_by_jd(df, job_description, k)

                st.success(f"Top {k} resumes shortlisted!")
                for rank, r in enumerate(top_resumes, 1):
                    with st.expander(f"Rank {rank}: {r['name']} - Score: {r['score']}/100"):
                        st.markdown(f"**Matched Keywords:** {', '.join(r['matched_keywords'])}")
                        st.markdown(f"**Resume Snippet:** {r['resume_snippet']}")
                        st.json(r['details'])

            except Exception as e:
                st.error(f"Error during processing: {e}")
