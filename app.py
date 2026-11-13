import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    r2_score,
    mean_absolute_error,
    silhouette_score,
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ICC T20 World Cup 2026 Analytics & Predictor",
    page_icon="🏏",
    layout="wide",
)

st.title("🏏 ICC Men's T20 World Cup 2026 — Predictive & Clustering Suite")
st.caption(
    "Business Analytics Group Project | Hosts: India & Sri Lanka (Feb 7 – Mar 8, 2026) | 55 Official Matches"
)

# -----------------------------------------------------------------------------
# MEMBER 1: DATA LOADING & FEATURE ENGINEERING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("icc_t20_wc_2026_matches.csv")
    completed = df[df["winner"] != "No Result"].copy()
    return df, completed


@st.cache_resource
def train_models(completed: pd.DataFrame):
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
            sym_rows.append(
                {
                    "rank_diff": o_rank - t_rank,
                    "team_rank": t_rank,
                    "opp_rank": o_rank,
                    "host_advantage": t_host - o_host,
                    "stage_num": s_num,
                    "won": won,
                    "dominance_index": r["dominance_index"]
                    if won == 1
                    else -r["dominance_index"],
                }
            )

    sym_df = pd.DataFrame(sym_rows)
    features = ["rank_diff", "team_rank", "opp_rank", "host_advantage", "stage_num"]
    X = sym_df[features]
    y_cls = sym_df["won"]
    y_reg = sym_df["dominance_index"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cls, test_size=0.25, random_state=42, stratify=y_cls
    )
    lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=120, max_depth=4, random_state=42).fit(
        X_train, y_train
    )

    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X, y_reg, test_size=0.25, random_state=42
    )
    ridge = Ridge(alpha=1.0).fit(X_tr_r, y_tr_r)

    metrics = {
        "lr_acc": accuracy_score(y_test, lr.predict(X_test)),
        "lr_auc": roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]),
        "rf_acc": accuracy_score(y_test, rf.predict(X_test)),
        "rf_auc": roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]),
        "lr_cm": confusion_matrix(y_test, lr.predict(X_test)),
        "rf_cm": confusion_matrix(y_test, rf.predict(X_test)),
        "ridge_r2": r2_score(y_te_r, ridge.predict(X_te_r)),
        "ridge_mae": mean_absolute_error(y_te_r, ridge.predict(X_te_r)),
        "y_te_r": y_te_r,
        "y_pred_r": ridge.predict(X_te_r),
    }
    return lr, rf, ridge, features, metrics


df, completed = load_data()
lr_model, rf_model, ridge_model, FEATURES, METRICS = train_models(completed)

# Build team-level lookup
teams = sorted(list(set(df["team_1"]).union(set(df["team_2"]))))
team_ranks = {}
for t in teams:
    if (df["team_1"] == t).any():
        team_ranks[t] = int(df[df["team_1"] == t]["team_1_icc_rank"].iloc[0])
    else:
        team_ranks[t] = int(df[df["team_2"] == t]["team_2_icc_rank"].iloc[0])

HOSTS = {"India", "Sri Lanka"}

# -----------------------------------------------------------------------------
# SIDEBAR: LIVE WHAT-IF MATCHUP SIMULATOR
# -----------------------------------------------------------------------------
st.sidebar.header("🎯 Live Matchup Simulator")
team_a = st.sidebar.selectbox("Select Team A", teams, index=teams.index("India"))
team_b = st.sidebar.selectbox("Select Team B", teams, index=teams.index("New Zealand"))
stage_choice = st.sidebar.selectbox(
    "Tournament Stage", ["Group Stage", "Super 8", "Semi-final", "Final"], index=3
)
model_choice = st.sidebar.radio(
    "Classification Model", ["Logistic Regression", "Random Forest"]
)

stage_map = {"Group Stage": 1, "Super 8": 2, "Semi-final": 3, "Final": 4}
rank_a, rank_b = team_ranks[team_a], team_ranks[team_b]
host_adv = (1 if team_a in HOSTS else 0) - (1 if team_b in HOSTS else 0)

input_row = pd.DataFrame(
    [
        {
            "rank_diff": rank_b - rank_a,
            "team_rank": rank_a,
            "opp_rank": rank_b,
            "host_advantage": host_adv,
            "stage_num": stage_map[stage_choice],
        }
    ]
)

active_model = lr_model if model_choice == "Logistic Regression" else rf_model
prob_a = float(active_model.predict_proba(input_row)[0][1])
prob_b = 1.0 - prob_a
pred_dom = float(ridge_model.predict(input_row)[0])

st.sidebar.markdown("---")
st.sidebar.subheader("Quick Prediction")
if team_a == team_b:
    st.sidebar.warning("Select two different teams to simulate a matchup.")
else:
    st.sidebar.metric(f"{team_a} Win Probability", f"{prob_a*100:.1f}%")
    st.sidebar.metric(f"{team_b} Win Probability", f"{prob_b*100:.1f}%")
    underdog = team_a if rank_a > rank_b else team_b
    underdog_prob = prob_a if rank_a > rank_b else prob_b
    if underdog_prob >= 0.35:
        st.sidebar.error(f"🚨 Upset Alert: {underdog} has {underdog_prob*100:.1f}% chance!")
    else:
        st.sidebar.success("✅ Low Upset Risk — Favorite strongly favored.")

# -----------------------------------------------------------------------------
# TABS (4-MEMBER STRUCTURE)
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 1. Tournament Pulse & Upsets",
        "🤖 2. Win & Upset Predictor",
        "📈 3. Dominance Forecaster",
        "🧬 4. Team DNA Clustering",
    ]
)

# --- TAB 1: EDA & UPSETS (MEMBER 2) ---
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Champions", "India (3rd Title)", "Beat NZ by 96 runs")
    c2.metric("Matches Played", f"{len(df)}", f"{len(completed)} Completed")
    c3.metric("Tournament Upsets", f"{int(completed['is_upset'].sum())}", "Lower-ranked wins")
    c4.metric("Player of the Series", "Sanju Samson", "Most Runs: S. Farhan (383)")

    st.subheader("Top Teams by Match Wins")
    win_counts = completed["winner"].value_counts().reset_index()
    win_counts.columns = ["Team", "Wins"]
    fig_wins = px.bar(
        win_counts.head(10),
        x="Wins",
        y="Team",
        orientation="h",
        color="Wins",
        color_continuous_scale="Tealgrn",
    )
    fig_wins.update_layout(yaxis=dict(autorange="reversed"), height=380)
    st.plotly_chart(fig_wins, use_container_width=True)

    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        st.subheader("🚨 Biggest Upsets in the 2026 Tournament")
        upsets_df = completed[completed["is_upset"] == 1][
            ["date", "stage", "team_1", "team_2", "winner", "margin_value", "margin_type"]
        ]
        st.dataframe(upsets_df, use_container_width=True, hide_index=True)
    with col_right:
        st.subheader("Stage-wise Match Breakdown")
        stage_counts = df["stage"].value_counts().reset_index()
        stage_counts.columns = ["Stage", "Matches"]
        fig_stage = px.pie(stage_counts, names="Stage", values="Matches", hole=0.45)
        st.plotly_chart(fig_stage, use_container_width=True)

# --- TAB 2: CLASSIFICATION & WIN PROBABILITY (MEMBER 3) ---
with tab2:
    st.subheader(f"Head-to-Head Win Probability: {team_a} vs {team_b} ({stage_choice})")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("LogReg Accuracy", f"{METRICS['lr_acc']*100:.1f}%")
    m2.metric("LogReg ROC-AUC", f"{METRICS['lr_auc']:.3f}")
    m3.metric("Random Forest Accuracy", f"{METRICS['rf_acc']*100:.1f}%")
    m4.metric("Random Forest ROC-AUC", f"{METRICS['rf_auc']:.3f}")

    gauge_fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_a * 100,
            title={"text": f"{team_a} Win Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 35], "color": "#fde2e4"},
                    {"range": [35, 65], "color": "#fff1cc"},
                    {"range": [65, 100], "color": "#d8f3dc"},
                ],
            },
        )
    )
    gauge_fig.update_layout(height=320)
    st.plotly_chart(gauge_fig, use_container_width=True)

    col_fi, col_cm = st.columns(2)
    with col_fi:
        st.subheader("Feature Importance (Random Forest)")
        fi_df = pd.DataFrame(
            {"Feature": FEATURES, "Importance": rf_model.feature_importances_}
        ).sort_values("Importance", ascending=True)
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig_fi, use_container_width=True)
    with col_cm:
        st.subheader(f"Confusion Matrix ({model_choice})")
        cm = METRICS["lr_cm"] if model_choice == "Logistic Regression" else METRICS["rf_cm"]
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=["Pred Loss", "Pred Win"],
            y=["Actual Loss", "Actual Win"],
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

# --- TAB 3: REGRESSION / DOMINANCE FORECASTER (MEMBER 3) ---
with tab3:
    st.subheader("Margin-Adjusted Dominance Index Forecaster (Ridge Regression)")
    r1, r2, r3 = st.columns(3)
    r1.metric("Ridge Regression R²", f"{METRICS['ridge_r2']:.3f}")
    r2.metric("Mean Absolute Error (MAE)", f"{METRICS['ridge_mae']:.1f} pts")
    r3.metric(
        f"Predicted Dominance ({team_a})",
        f"{pred_dom:+.1f} / 100",
        "Positive favors Team A",
    )

    st.write(
        "The **Dominance Index (0–100)** normalizes victories by runs (capped at 100) and wickets (wickets × 10) "
        "so that 10-wicket wins and 96-run wins can be modeled on a unified scale."
    )

    reg_df = pd.DataFrame(
        {"Actual Dominance": METRICS["y_te_r"], "Predicted Dominance": METRICS["y_pred_r"]}
    )
    fig_reg = px.scatter(
        reg_df,
        x="Actual Dominance",
        y="Predicted Dominance",
        trendline="ols",
        title="Actual vs. Predicted Match Dominance (Test Set)",
    )
    st.plotly_chart(fig_reg, use_container_width=True)

# --- TAB 4: K-MEANS TEAM DNA CLUSTERING (MEMBER 4) ---
with tab4:
    st.subheader("Team DNA Segmentation (K-Means Clustering across all 20 Nations)")
    k_clusters = st.slider("Number of Clusters (k)", min_value=2, max_value=5, value=4)

    team_rows = []
    for t in teams:
        sub = completed[(completed["team_1"] == t) | (completed["team_2"] == t)]
        wins = int((sub["winner"] == t).sum())
        win_pct = round(wins / len(sub) * 100, 1) if len(sub) > 0 else 0.0
        upsets = int(sub[(sub["winner"] == t) & (sub["is_upset"] == 1)].shape[0])
        avg_dom = sub[sub["winner"] == t]["dominance_index"].mean()
        team_rows.append(
            {
                "Team": t,
                "ICC Rank": team_ranks[t],
                "Matches": len(sub),
                "Wins": wins,
                "Win %": win_pct,
                "Upsets Caused": upsets,
                "Avg Win Dominance": 0.0 if pd.isna(avg_dom) else round(float(avg_dom), 1),
            }
        )

    t_df = pd.DataFrame(team_rows)
    scaler = StandardScaler()
    X_clust = scaler.fit_transform(
        t_df[["ICC Rank", "Win %", "Avg Win Dominance", "Upsets Caused"]]
    )
    km = KMeans(n_clusters=k_clusters, random_state=42, n_init=10).fit(X_clust)
    t_df["Cluster"] = km.labels_.astype(str)
    sil = silhouette_score(X_clust, km.labels_)

    st.metric("Clustering Silhouette Score", f"{sil:.3f}", "Higher = tighter clusters")

    fig_clust = px.scatter(
        t_df,
        x="ICC Rank",
        y="Win %",
        size="Avg Win Dominance",
        color="Cluster",
        hover_name="Team",
        text="Team",
        title="Team DNA: ICC Ranking vs. 2026 Tournament Win % (Bubble Size = Dominance)",
    )
    fig_clust.update_traces(textposition="top center")
    fig_clust.update_layout(height=480)
    st.plotly_chart(fig_clust, use_container_width=True)

    st.dataframe(
        t_df.sort_values(["Cluster", "Win %"], ascending=[True, False]),
        use_container_width=True,
        hide_index=True,
    )