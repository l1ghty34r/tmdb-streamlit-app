# 🎬 TMDB Movie Analytics App

Interactive Streamlit app for analyzing movie data, with a focus on exploration, recommendation systems, and ROI evaluation.

👉 **Live Demo:**  
https://tmdb-app-app-7fq2yqkxrpknmdkg4b7nqq.streamlit.app/

---

## 🚀 Features

### 🔍 Movie Database
- Search and filter movies (genre, cast, director, etc.)
- Detailed view with all relevant information
- Clear presentation of reviews and metadata

### 🎯 Recommendation Engine
- Content-based recommender based on movie metadata  
- Switched from CountVectorizer to **TF-IDF** to reduce less relevant terms and improve recommendation quality  
- Computes similarities between movies to generate recommendations  

### 📈 ROI Builder
- Calculates an expected ROI based on:
  - Actors  
  - Directors  
  - Genres  
- Goal: provide a data-driven estimation of potentially successful combinations  
- Note: Marketing costs are not included → ROI tends to be higher than in reality  

---

## 🧠 Tech Stack

- Python (Pandas, NumPy, Scikit-learn)  
- SQL / PostgreSQL (hosted on Neon)  
- Streamlit (frontend & deployment)  

**Machine Learning (lightweight):**
- TF-IDF for feature weighting  
- Cosine similarity for recommendations  

---

## 🗄️ Data Source

- Dataset: *The Movie Database (TMDB) – Kaggle*  

**Data processing included:**
- Cleaning (missing values, data types, etc.)  
- Normalization (custom core tables in PostgreSQL)  
- Optimization for analysis and app usage  

---

## ⚙️ Setup (Local)

```bash
git clone <your-repo-link>
cd tmdb-streamlit-app
pip install -r requirements.txt
streamlit run app.py
