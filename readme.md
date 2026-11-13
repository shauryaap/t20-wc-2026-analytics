# ICC Men's T20 World Cup 2026 — Predictive & Clustering Analytics App

Business Analytics Group Project built with **Google Colab**, hosted on **GitHub**, and deployed on **Streamlit Community Cloud**.

## Verified Data Sources
- **Match Results (All 55 Matches, Feb 7 – Mar 8, 2026):** [Sky Sports Full Results](https://www.skysports.com/cricket/news/13502048/mens-t20-world-cup-2026-fixtures-england-india-pakistan-australia-scotland-ireland-west-indies-and-more) and [Wikipedia — 2026 Men's T20 World Cup](https://en.wikipedia.org/wiki/2026_Men%27s_T20_World_Cup)
- **Additional Ball-by-Ball / Historical Datasets:** [Cricsheet Downloads](https://cricsheet.org/downloads/) | [Kaggle T20 WC 2007–2026 Dataset](https://www.kaggle.com/datasets/abdulmaliklodhra/t20-world-cup-data-analysis-2007-2026)

## Files in This Repository
- `icc_t20_wc_2026_matches.csv` — Clean 55-match dataset with ICC pre-tournament rankings, host flags, margin types, Dominance Index (0–100), and upset flags.
- `T20_WC_2026_Colab_Pipeline.ipynb` — Google Colab notebook for EDA, model training, and evaluation.
- `app.py` — Interactive Streamlit web application with 4 analytical modules.
- `requirements.txt` — Python dependencies for Streamlit Cloud.

## How to Deploy on GitHub + Streamlit Cloud (3 Minutes)
1. Create a public GitHub repository (e.g. `t20-wc-2026-analytics`) and upload all 4 files (`app.py`, `icc_t20_wc_2026_matches.csv`, `requirements.txt`, `T20_WC_2026_Colab_Pipeline.ipynb`).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and click **New app**.
3. Select your repository, branch (`main`), and main file path (`app.py`), then click **Deploy**.

## 4-Member Presentation & Viva Split
1. **Member 1 (Data & Feature Engineering):** Explains `load_data()`, symmetric match-perspective encoding (`rank_diff`, `host_advantage`, `stage_num`), and why `No Result` matches are filtered before training.
2. **Member 2 (EDA & Upset Analysis — Tab 1):** Explains tournament KPIs, top winning teams, and notable upsets (e.g., Zimbabwe beating Australia, Italy beating Nepal).
3. **Member 3 (Predictive Modeling — Tabs 2 & 3):** Explains Logistic Regression vs. Random Forest (`predict_proba`, Accuracy, ROC-AUC, Confusion Matrix) and Ridge Regression for Dominance Index forecasting (R², MAE).
4. **Member 4 (Clustering & Deployment — Tab 4 & Sidebar):** Explains K-Means clustering (`StandardScaler`, Silhouette Score, k=4 team tiers), `@st.cache_resource`, and how GitHub connects to Streamlit Community Cloud.
