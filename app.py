import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve,
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- Global ---- */
[data-testid="stAppViewContainer"] {
    background: #f0f4f8;
}
[data-testid="stSidebar"] {
    background: #1e293b !important;
    border-right: none;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }
[data-testid="stSidebarContent"] label { color: #94a3b8 !important; }

/* ---- KPI tiles ---- */
.kpi-tile {
    background: white;
    border-radius: 14px;
    padding: 22px 18px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border-top: 4px solid var(--accent, #6366f1);
}
.kpi-num  { font-size: 2rem; font-weight: 800; color: #1e293b; line-height: 1.1; }
.kpi-sub  { font-size: 0.78rem; color: #64748b; margin-top: 6px; }

/* ---- Winner banner ---- */
.winner-banner {
    background: linear-gradient(90deg, #eff6ff, #f5f3ff);
    border: 1.5px solid #818cf8;
    border-radius: 12px;
    padding: 14px 20px;
    color: #3730a3;
    font-size: 1.02rem;
}

/* ---- Section pills ---- */
.pill {
    display: inline-block;
    background: #e0e7ff;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.8rem;
    color: #4338ca;
    margin: 2px;
    font-weight: 500;
}

/* ---- Subtle divider ---- */
.sec-div { border-top: 1px solid #e2e8f0; margin: 24px 0; }

/* ---- Main content card feel ---- */
[data-testid="stVerticalBlock"] > div { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_DEFS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes":         GaussianNB(),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
}

PALETTE = {
    "Logistic Regression": "#6366f1",   # indigo
    "Decision Tree":       "#10b981",   # emerald
    "KNN":                 "#f59e0b",   # amber
    "Naive Bayes":         "#ec4899",   # pink
    "Random Forest":       "#3b82f6",   # blue
}

KPI_ACCENTS = ["#6366f1", "#10b981", "#f59e0b", "#3b82f6"]

def hex_rgba(hex_color, alpha=0.2):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

TRAIN_CSV = "train_data.csv"

# ── Train (cached) ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training all models on training data…")
def train_all():
    df   = pd.read_csv(TRAIN_CSV)
    X    = df.drop("target", axis=1)
    y    = df["target"]
    sc   = StandardScaler()
    Xs   = sc.fit_transform(X)
    trained = {}
    for name, mdl in MODEL_DEFS.items():
        mdl.fit(Xs, y)
        trained[name] = mdl
    return trained, sc, list(X.columns)

# ── Evaluate one model ─────────────────────────────────────────────────────────
def evaluate(model, scaler, X_test, y_test):
    Xs     = scaler.transform(X_test)
    y_pred = model.predict(Xs)
    y_prob = (model.predict_proba(Xs)[:, 1]
              if hasattr(model, "predict_proba") else y_pred.astype(float))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    return {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "AUC":       round(roc_auc_score(y_test, y_prob),  4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred),   4),
        "F1":        round(f1_score(y_test, y_pred),       4),
        "MCC":       round(matthews_corrcoef(y_test, y_pred), 4),
        "_y_pred":   y_pred,
        "_y_prob":   y_prob,
        "_fpr":      fpr,
        "_tpr":      tpr,
        "_report":   classification_report(y_test, y_pred,
                         target_names=["Malignant (0)", "Benign (1)"],
                         output_dict=True),
    }

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 ML Dashboard")
    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "**Upload Test Data (CSV)**",
        type="csv",
        help="CSV must have the same 30 feature columns + a 'target' column.",
    )

    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "**Select Model**",
        ["All Models"] + list(MODEL_DEFS.keys()),
    )

    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)
    st.markdown("""
**Dataset Info**
<span class='pill'>Breast Cancer Wisconsin</span>
<span class='pill'>UCI / sklearn</span>

| Property | Value |
|----------|-------|
| Instances | 569 |
| Features | 30 |
| Classes | 2 |
| Train split | 455 |
| Test split | 114 |
""", unsafe_allow_html=True)

    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)
    st.caption("BITS Pilani WILP · M.Tech AIML/DSE  \nMachine Learning · Assignment 2")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🧬 ML Classification Dashboard")
st.markdown(
    "**Breast Cancer Wisconsin** — Binary classification comparing 5 ML models "
    "across 6 evaluation metrics."
)
st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)

# ── Load test data ──────────────────────────────────────────────────────────────
if uploaded is not None:
    test_df = pd.read_csv(uploaded)
    st.success(f"Uploaded **{uploaded.name}** — {test_df.shape[0]} rows, {test_df.shape[1]} cols")
elif os.path.exists("test_data.csv"):
    test_df = pd.read_csv("test_data.csv")
    st.info("Using bundled **test_data.csv**. Upload a different CSV via the sidebar.")
else:
    st.error("No test data found. Please upload a CSV file.")
    st.stop()

if "target" not in test_df.columns:
    st.error("CSV must contain a **'target'** column.")
    st.stop()

y_test = test_df["target"]
X_test = test_df.drop("target", axis=1)

# ── Train / load models ────────────────────────────────────────────────────────
if not os.path.exists(TRAIN_CSV):
    st.error(f"Training file **{TRAIN_CSV}** not found.")
    st.stop()

trained_models, scaler, train_feats = train_all()
try:
    X_test = X_test[train_feats]
except KeyError as e:
    st.error(f"Test data missing feature column(s): {e}")
    st.stop()

# ── Run evaluation ─────────────────────────────────────────────────────────────
models_to_show = list(MODEL_DEFS.keys()) if model_choice == "All Models" else [model_choice]
results = {name: evaluate(trained_models[name], scaler, X_test, y_test)
           for name in models_to_show}

metric_keys = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
metrics_df  = pd.DataFrame(
    [{**{"Model": n}, **{k: results[n][k] for k in metric_keys}}
     for n in models_to_show]
).set_index("Model")

best_model = metrics_df["Accuracy"].idxmax()

# ── KPI row ────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
best = results[best_model]
c1.markdown(f"""<div class="kpi-tile" style="--accent:#6366f1">
  <div class="kpi-num" style="color:#6366f1">{best["Accuracy"]:.2%}</div>
  <div class="kpi-sub">Best Accuracy &nbsp;·&nbsp; {best_model}</div>
</div>""", unsafe_allow_html=True)
c2.markdown(f"""<div class="kpi-tile" style="--accent:#10b981">
  <div class="kpi-num" style="color:#10b981">{best["AUC"]:.4f}</div>
  <div class="kpi-sub">Best AUC &nbsp;·&nbsp; {best_model}</div>
</div>""", unsafe_allow_html=True)
c3.markdown(f"""<div class="kpi-tile" style="--accent:#f59e0b">
  <div class="kpi-num" style="color:#f59e0b">{best["F1"]:.4f}</div>
  <div class="kpi-sub">Best F1 Score &nbsp;·&nbsp; {best_model}</div>
</div>""", unsafe_allow_html=True)
c4.markdown(f"""<div class="kpi-tile" style="--accent:#3b82f6">
  <div class="kpi-num" style="color:#3b82f6">{len(y_test)}</div>
  <div class="kpi-sub">Test Samples</div>
</div>""", unsafe_allow_html=True)

st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Metrics Comparison",
    "📈 ROC Curves",
    "🔲 Confusion Matrices",
    "🔍 Deep Dive",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Metrics Comparison
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    # Styled metrics table
    st.subheader("Evaluation Metrics Table")

    def highlight_best(s):
        best_val = s.max()
        return [
            "background-color:#dcfce7; color:#166534; font-weight:700"
            if v == best_val else ""
            for v in s
        ]

    st.dataframe(
        metrics_df.style.apply(highlight_best, axis=0).format("{:.4f}"),
        use_container_width=True,
    )

    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)

    # Grouped bar chart — all metrics side by side
    st.subheader("Metric Comparison — All Models")
    fig_bar = go.Figure()
    for name in models_to_show:
        fig_bar.add_trace(go.Bar(
            name=name,
            x=metric_keys,
            y=[results[name][k] for k in metric_keys],
            marker_color=PALETTE.get(name, "#58a6ff"),
            text=[f"{results[name][k]:.3f}" for k in metric_keys],
            textposition="outside",
            textfont_size=11,
        ))
    fig_bar.update_layout(
        barmode="group",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="#1e293b",
        yaxis=dict(range=[0, 1.18], gridcolor="#e2e8f0"),
        xaxis=dict(gridcolor="#e2e8f0"),
        legend=dict(orientation="h", y=1.12, bgcolor="white"),
        height=420,
        margin=dict(t=50, b=30),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)

    # Radar chart
    if len(models_to_show) > 1:
        st.subheader("Radar Chart — Model Profiles")
        radar_cats = metric_keys + [metric_keys[0]]  # close the polygon
        fig_radar = go.Figure()
        for name in models_to_show:
            vals = [results[name][k] for k in metric_keys]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=radar_cats,
                name=name,
                fill="toself",
                fillcolor=hex_rgba(PALETTE.get(name, "#58a6ff"), 0.2),
                line=dict(color=PALETTE.get(name, "#58a6ff"), width=2),
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#f8fafc",
                radialaxis=dict(visible=True, range=[0, 1],
                                gridcolor="#cbd5e1", color="#475569"),
                angularaxis=dict(gridcolor="#cbd5e1", color="#475569"),
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_color="#1e293b",
            legend=dict(orientation="h", y=-0.12, bgcolor="white"),
            height=440,
            margin=dict(t=30, b=60),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Winner callout
    st.markdown(f"""
<div class="winner-banner">
🏆 <strong>Best Model on this test set:</strong> {best_model} &nbsp;|&nbsp;
Accuracy: {results[best_model]["Accuracy"]:.4f} &nbsp;|&nbsp;
AUC: {results[best_model]["AUC"]:.4f} &nbsp;|&nbsp;
F1: {results[best_model]["F1"]:.4f} &nbsp;|&nbsp;
MCC: {results[best_model]["MCC"]:.4f}
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ROC Curves
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("ROC Curves")
    fig_roc = go.Figure()
    # Chance line
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="#94a3b8", width=1.5),
        name="Random Chance",
        showlegend=True,
    ))
    for name in models_to_show:
        r = results[name]
        fig_roc.add_trace(go.Scatter(
            x=r["_fpr"], y=r["_tpr"],
            mode="lines",
            name=f"{name} (AUC={r['AUC']:.4f})",
            line=dict(color=PALETTE.get(name, "#58a6ff"), width=2.5),
        ))
    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="#1e293b",
        xaxis=dict(gridcolor="#e2e8f0", range=[0, 1]),
        yaxis=dict(gridcolor="#e2e8f0", range=[0, 1.02]),
        legend=dict(x=0.55, y=0.12, bgcolor="white",
                    bordercolor="#e2e8f0", borderwidth=1),
        height=500,
        margin=dict(t=30),
    )
    fig_roc.add_annotation(
        text="↑ Better",
        x=0.08, y=0.93,
        showarrow=False,
        font=dict(size=11, color="#64748b"),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # AUC summary
    st.subheader("AUC Scores")
    auc_vals = {n: results[n]["AUC"] for n in models_to_show}
    sorted_auc = sorted(auc_vals.items(), key=lambda x: x[1], reverse=True)
    fig_auc = go.Figure(go.Bar(
        x=[v for _, v in sorted_auc],
        y=[n for n, _ in sorted_auc],
        orientation="h",
        marker_color=[PALETTE.get(n, "#58a6ff") for n, _ in sorted_auc],
        text=[f"{v:.4f}" for _, v in sorted_auc],
        textposition="outside",
        textfont_size=12,
    ))
    fig_auc.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="#1e293b",
        xaxis=dict(range=[0, 1.12], gridcolor="#e2e8f0"),
        yaxis=dict(gridcolor="#e2e8f0"),
        height=300,
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_auc, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Confusion Matrices
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Confusion Matrices")
    n = len(models_to_show)
    cols = st.columns(min(n, 3))

    for idx, name in enumerate(models_to_show):
        col = cols[idx % 3]
        with col:
            cm = confusion_matrix(y_test, results[name]["_y_pred"])
            # Normalised (%) for annotation
            cm_norm = cm.astype(float) / cm.sum()

            fig_cm, ax = plt.subplots(figsize=(4, 3.5))
            fig_cm.patch.set_facecolor("white")
            ax.set_facecolor("white")

            annot = np.array([[f"{cm[i,j]}\n({cm_norm[i,j]:.1%})"
                               for j in range(2)] for i in range(2)])
            sns.heatmap(
                cm, annot=annot, fmt="", cmap="Blues", ax=ax,
                xticklabels=["Malignant", "Benign"],
                yticklabels=["Malignant", "Benign"],
                linewidths=0.5, linecolor="#e2e8f0",
                cbar_kws={"shrink": 0.8},
            )
            ax.set_title(name, color="#1e293b", fontsize=11, fontweight="bold", pad=8)
            ax.set_xlabel("Predicted", color="#475569", fontsize=9)
            ax.set_ylabel("Actual",    color="#475569", fontsize=9)
            ax.tick_params(colors="#475569", labelsize=8)
            plt.tight_layout()
            st.pyplot(fig_cm, use_container_width=True)
            plt.close(fig_cm)

    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)

    # Classification report
    st.subheader("Classification Reports")
    for name in models_to_show:
        with st.expander(f"Report — {name}"):
            rpt = results[name]["_report"]
            rpt_df = pd.DataFrame(rpt).T
            rpt_df = rpt_df[["precision", "recall", "f1-score", "support"]].round(4)
            st.dataframe(rpt_df, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Deep Dive
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    left, right = st.columns([1, 1])

    # Feature importance (tree-based models)
    with left:
        st.subheader("Feature Importance")
        fi_models = [n for n in models_to_show
                     if hasattr(trained_models[n], "feature_importances_")]
        if fi_models:
            fi_choice = st.selectbox("Model", fi_models, key="fi_sel")
            importances = trained_models[fi_choice].feature_importances_
            fi_df = pd.DataFrame({"Feature": train_feats, "Importance": importances})
            fi_df = fi_df.sort_values("Importance", ascending=True).tail(15)

            fig_fi = go.Figure(go.Bar(
                x=fi_df["Importance"],
                y=fi_df["Feature"],
                orientation="h",
                marker_color=PALETTE.get(fi_choice, "#58a6ff"),
            ))
            fig_fi.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font_color="#1e293b",
                xaxis=dict(gridcolor="#e2e8f0"),
                yaxis=dict(gridcolor="#e2e8f0", tickfont_size=10),
                height=450,
                margin=dict(t=10, l=10),
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("No tree-based model selected. Choose Decision Tree or Random Forest.")

    # Logistic Regression coefficients
    with right:
        st.subheader("LR Coefficient Magnitudes")
        if "Logistic Regression" in models_to_show:
            coef = trained_models["Logistic Regression"].coef_[0]
            coef_df = pd.DataFrame({"Feature": train_feats, "Coefficient": coef})
            coef_df["Abs"] = coef_df["Coefficient"].abs()
            coef_df = coef_df.sort_values("Abs", ascending=True).tail(15)
            colors = ["#ef4444" if v < 0 else "#10b981"
                      for v in coef_df["Coefficient"]]
            fig_coef = go.Figure(go.Bar(
                x=coef_df["Coefficient"],
                y=coef_df["Feature"],
                orientation="h",
                marker_color=colors,
            ))
            fig_coef.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font_color="#1e293b",
                xaxis=dict(gridcolor="#e2e8f0"),
                yaxis=dict(gridcolor="#e2e8f0", tickfont_size=10),
                height=450,
                margin=dict(t=10, l=10),
            )
            fig_coef.add_vline(x=0, line_color="#94a3b8", line_width=1)
            st.plotly_chart(fig_coef, use_container_width=True)
        else:
            st.info("Select 'Logistic Regression' or 'All Models' to see coefficients.")

    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)

    # Raw predictions table
    st.subheader("Prediction Preview (first 30 rows)")
    preview = X_test.copy()
    preview.insert(0, "Actual", y_test.values)
    for name in models_to_show:
        preview[f"{name[:6]}…Pred"] = results[name]["_y_pred"]
    st.dataframe(preview.head(30), use_container_width=True)

    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)

    # Dataset class distribution
    st.subheader("Test Set Class Distribution")
    vc = y_test.value_counts().reset_index()
    vc.columns = ["Class", "Count"]
    vc["Label"] = vc["Class"].map({0: "Malignant", 1: "Benign"})
    fig_dist = px.pie(
        vc, values="Count", names="Label",
        color_discrete_sequence=["#ef4444", "#10b981"],
        hole=0.45,
    )
    fig_dist.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="#1e293b",
        height=300,
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)
st.caption("BITS Pilani WILP · M.Tech AIML/DSE · Machine Learning · Assignment 2  "
           "| Dataset: Breast Cancer Wisconsin (UCI)")
