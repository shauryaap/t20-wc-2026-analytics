# %% [markdown]
# # ICC Men's T20 World Cup 2026 — Predictive & Clustering Analytics
# **Business Analytics Group Project** | Built in Google Colab, Deployed via GitHub + Streamlit

# %%
# MEMBER 1: Data Loading & Preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Upload icc_t20_wc_2026_matches.csv to Colab or replace with your GitHub raw CSV URL
DATA_URL = "icc_t20_wc_2026_matches.csv"
df = pd.read_csv(DATA_URL)
print("Dataset Shape:", df.shape)
df.head()

# %%
# MEMBER 2: Exploratory Data Analysis (EDA) & Upset Discovery
completed = df[df["winner"] != "No Result"].copy()
print(f"Completed Matches: {len(completed)} | Total Upsets: {completed['is_upset'].sum()}")
print("\nBiggest Upsets in 2026 T20 World Cup:")
print(completed[completed["is_upset"] == 1][["date", "stage", "team_1", "team_2", "winner", "margin_value", "margin_type"]])

plt.figure(figsize=(10, 4))
top_winners = completed["winner"].value_counts().head(8)
sns.barplot(x=top_winners.values, y=top_winners.index, palette="viridis")
plt.title("Top Teams by Match Wins — ICC T20 World Cup 2026")
plt.xlabel("Wins")
plt.show()

# %%
# MEMBER 3: Match Winner Classification & Dominance Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, r2_score, mean_absolute_error

stage_map = {"Group Stage": 1, "Super 8": 2, "Semi-final": 3, "Final": 4}
sym_rows = []
for _, r in completed.iterrows():
    s_num = stage_map[r["stage"]]
    for side in [1, 2]:
        t_rank = r["team_1_icc_rank"] if side == 1 else r["team_2_icc_rank"]
        o_rank = r["team_2_icc_rank"] if side == 1 else r["team_1_icc_rank"]
        t_host = r["team_1_is_host"] if side == 1 else r["team_2_is_host"]
        o_host = r["team_2_is_host"] if side == 1 else r["team_1_is_host"]
        won = 1 if r["winner"] == (r["team_1"] if side == 1 else r["team_2"]) else 0
        sym_rows.append({
            "rank_diff": o_rank - t_rank,
            "team_rank": t_rank,
            "opp_rank": o_rank,
            "host_advantage": t_host - o_host,
            "stage_num": s_num,
            "won": won,
            "dominance_index": r["dominance_index"] if won == 1 else -r["dominance_index"],
        })

sym_df = pd.DataFrame(sym_rows)
features = ["rank_diff", "team_rank", "opp_rank", "host_advantage", "stage_num"]
X = sym_df[features]
y_cls = sym_df["won"]

X_train, X_test, y_train, y_test = train_test_split(X, y_cls, test_size=0.25, random_state=42, stratify=y_cls)
lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42).fit(X_train, y_train)

print("Logistic Regression -> Accuracy:", round(accuracy_score(y_test, lr.predict(X_test)), 3), "| ROC-AUC:", round(roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]), 3))
print("Random Forest       -> Accuracy:", round(accuracy_score(y_test, rf.predict(X_test)), 3), "| ROC-AUC:", round(roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]), 3))

X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X, sym_df["dominance_index"], test_size=0.25, random_state=42)
ridge = Ridge(alpha=1.0).fit(X_tr_r, y_tr_r)
print("Ridge Dominance     -> R2:", round(r2_score(y_te_r, ridge.predict(X_te_r)), 3), "| MAE:", round(mean_absolute_error(y_te_r, ridge.predict(X_te_r)), 2))

# %%
# MEMBER 4: Team DNA Clustering (K-Means)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

teams = sorted(list(set(df["team_1"]).union(set(df["team_2"]))))
team_rows = []
for team in teams:
    sub = completed[(completed["team_1"] == team) | (completed["team_2"] == team)]
    rank = df[df["team_1"] == team]["team_1_icc_rank"].iloc[0] if (df["team_1"] == team).any() else df[df["team_2"] == team]["team_2_icc_rank"].iloc[0]
    wins = (sub["winner"] == team).sum()
    win_pct = wins / len(sub) * 100 if len(sub) > 0 else 0
    upsets = sub[(sub["winner"] == team) & (sub["is_upset"] == 1)].shape[0]
    avg_dom = sub[sub["winner"] == team]["dominance_index"].mean()
    team_rows.append({
        "team": team,
        "icc_rank": rank,
        "wins": wins,
        "win_pct": round(win_pct, 1),
        "upsets_caused": upsets,
        "avg_win_dominance": 0.0 if pd.isna(avg_dom) else round(avg_dom, 1),
    })

t_df = pd.DataFrame(team_rows)
scaler = StandardScaler()
X_c = scaler.fit_transform(t_df[["icc_rank", "win_pct", "avg_win_dominance", "upsets_caused"]])
km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_c)
t_df["cluster"] = km.labels_
print("Silhouette Score:", round(silhouette_score(X_c, km.labels_), 3))
print(t_df.sort_values("win_pct", ascending=False))