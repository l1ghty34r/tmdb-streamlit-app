🎬 TMDB Movie Analytics App

Interaktive Streamlit-App zur Analyse von Filmdaten mit Fokus auf Exploration, Empfehlungssysteme und ROI-Bewertung.

👉 Live Demo: [Hier deinen Streamlit-Link einfügen]

🚀 Features

🔍 Movie Database
Suche und Filter nach Filmen (Genre, Cast, Director etc.)
Detailansicht mit allen relevanten Informationen
Übersichtliche Darstellung von Reviews und Metadaten

🎯 Recommendation Engine
Content-basierter Recommender auf Basis von Film-Metadaten
Umstellung von CountVectorizer auf TF-IDF, um irrelevante Begriffe zu reduzieren und die Qualität der Empfehlungen zu verbessern
Berechnung von Ähnlichkeiten zwischen Filmen zur Generierung von Empfehlungen

📈 ROI Builder
Berechnung eines erwarteten ROI basierend auf:
Schauspielern
Regisseuren
Genres
Ziel: datenbasierte Einschätzung, welche Kombinationen potenziell erfolgreich sein könnten
Hinweis: Marketingkosten sind nicht im Budget enthalten → ROI tendenziell höher als in der Realität

🧠 Tech Stack
Python (Pandas, NumPy, Scikit-learn)
SQL / PostgreSQL (gehostet über Neon)
Streamlit (Frontend & Deployment)
Machine Learning (leichtgewichtig):
TF-IDF zur Feature-Gewichtung
Cosine Similarity für Empfehlungen

🗄️ Datenbasis
Datensatz: The Movie Database (TMDB) – Kaggle
Daten wurden:
bereinigt (Missing Values, Datentypen etc.)
normalisiert (eigene Core-Tabellen in PostgreSQL)
für Analyse und App-Nutzung optimiert

⚙️ Setup (lokal)
git clone <dein-repo-link>
cd tmdb-streamlit-app
pip install -r requirements.txt
streamlit run app.py

🧩 Projektstruktur
tmdb-streamlit-app/
│
├── app.py
├── pages/
├── notebooks/        # Datenbereinigung & Exploration
├── requirements.txt
├── README.md

🎯 Ziel des Projekts

Ziel war es, eine End-to-End Data Application zu entwickeln:

Datenaufbereitung (Python + SQL)
Datenmodellierung (PostgreSQL)
Analyse & Feature Engineering
Umsetzung als interaktive Anwendung (Streamlit)
💡 Weiterentwicklung (optional)
Erweiterung um echte Machine-Learning-Modelle (z. B. Regression für ROI)
Integration von User-Ratings für Hybrid-Recommender
Deployment mit Docker
