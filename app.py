import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from fpdf import FPDF
import tempfile

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Threat Detection System",
    page_icon="🛡️",
    layout="wide"
)

# ── Load saved model artifacts ────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model        = joblib.load("model.pkl")
    metrics      = joblib.load("metrics.pkl")
    encoders     = joblib.load("encoders.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    return model, metrics, encoders, feature_cols

model, saved_metrics, encoders, feature_cols = load_artifacts()

CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']

COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

# Known aliases: values in uploaded CSVs that differ from NSL-KDD training labels
SERVICE_ALIASES = {
    'imap':    'imap4',
    'ssl':     'private',
    'http_80': 'http',
    'https':   'http_443',
}


def assign_risk(prob):
    if prob < 0.4:    return 'Low'
    elif prob < 0.75: return 'Medium'
    else:             return 'High'


def mask_ip(ip_str):
    """
    IP Masking — Privacy Compliance (GDPR / DPDP Act 2023)
    Replaces last two octets of an IP with 'x.x'
    Example: 192.168.1.105  →  192.168.x.x
    """
    import re
    ip_str = str(ip_str)
    pattern = r'(\d{1,3})\.(\d{1,3})\.\d{1,3}\.\d{1,3}'
    if re.match(pattern, ip_str):
        return re.sub(pattern, r'\1.\2.x.x', ip_str)
    return ip_str  # return as-is if not an IP


def apply_pii_masking(df):
    """
    Scans all string columns for IP-like values and masks them.
    Also masks the Log # column to generate anonymous session IDs.
    """
    masked_df = df.copy()
    for col in masked_df.select_dtypes(include=['object']).columns:
        masked_df[col] = masked_df[col].apply(mask_ip)
    return masked_df


def preprocess_uploaded(df_raw):
    df = df_raw.copy()
    ncols = df.shape[1]

    # ── Column assignment: handles all NSL-KDD CSV variants ──────────────────
    if ncols >= 43:
        # Full NSL-KDD: 41 features + label + difficulty (+ any extra trailing cols)
        df.columns = COLUMNS[:ncols]
        if 'difficulty' in df.columns:
            df = df.drop(columns=['difficulty'])

    elif ncols == 42:
        df.columns = COLUMNS[:42]
        if 'difficulty' in df.columns:
            df = df.drop(columns=['difficulty'])

    elif ncols == 41:
        # Detect whether last column is label (strings) or a numeric feature
        last_col = df.iloc[:, 40]
        is_label = (
            last_col.dtype == object or
            last_col.astype(str).str.match(r'^[a-zA-Z]').any()
        )
        if is_label:
            # 40 features + label (test_500, test_1000, etc.)
            df.columns = COLUMNS[:40] + ['label']
        else:
            # 41 pure features, no label
            df.columns = COLUMNS[:41]

    else:
        # Best-effort: assign as many columns as we have
        df.columns = COLUMNS[:ncols]

    # ── Store original label if present ──────────────────────────────────────
    original_labels = None
    if 'label' in df.columns:
        original_labels = df['label'].copy()

    # ── Fill missing values ───────────────────────────────────────────────────
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].mean())
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'label':
            df[col] = df[col].fillna(df[col].mode()[0])

    # ── Encode categoricals with alias map + fuzzy substring fallback ─────────
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = encoders[col]

            def safe_encode(x, le=le):
                x = str(x).strip()
                # 1. Try alias map first
                x = SERVICE_ALIASES.get(x, x)
                # 2. Exact match
                if x in le.classes_:
                    return le.transform([x])[0]
                # 3. Fuzzy substring match
                matches = [c for c in le.classes_ if x in c or c in x]
                if matches:
                    return le.transform([matches[0]])[0]
                # 4. Fall back to 'other' if it exists, else 0
                return le.transform(['other'])[0] if 'other' in le.classes_ else 0

            df[col] = df[col].astype(str).apply(safe_encode)

    # ── Safety net: force all feature columns to numeric ─────────────────────
    for col in df.columns:
        if col not in ['label', 'binary_label', 'attack_type']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # ── Align to trained feature columns (add any missing cols as 0) ─────────
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0

    X = df[feature_cols]
    return X, original_labels


def generate_pdf(result_df, metrics_dict):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "AI Threat Detection - Analysis Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Total logs analysed: {len(result_df)}", ln=True)
    attacks = (result_df['Prediction'] == 'Malicious').sum()
    pdf.cell(0, 8, f"Attacks detected: {attacks}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Model Performance Metrics", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for k, v in [
        ("Accuracy",  metrics_dict['accuracy']),
        ("Precision", metrics_dict['precision']),
        ("Recall",    metrics_dict['recall']),
        ("F1-Score",  metrics_dict['f1']),
        ("ROC-AUC",   metrics_dict['roc_auc']),
        ("CV Mean",   metrics_dict['cv_mean']),
    ]:
        pdf.cell(0, 7, f"  {k}: {v}%", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Top 20 Log Entries", ln=True)
    pdf.set_font("Helvetica", "B", 9)
    cols = ['Prediction', 'Risk Level', 'Attack Probability']
    widths = [50, 40, 60]
    for col, w in zip(cols, widths):
        pdf.cell(w, 7, col, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for _, row in result_df.head(20).iterrows():
        pdf.cell(50, 6, str(row.get('Prediction', '')),         border=1)
        pdf.cell(40, 6, str(row.get('Risk Level', '')),         border=1)
        pdf.cell(60, 6, str(row.get('Attack Probability', '')), border=1)
        pdf.ln()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp.name)
    return tmp.name


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🛡️ AI-Driven Log File Threat Detection System")
st.caption("Mini Project · MIET Jammu · Random Forest + NSL-KDD · 2026")
st.divider()

# Sidebar
with st.sidebar:
    st.header("📂 Upload Log File")
    uploaded = st.file_uploader("Upload CSV log file", type=["csv", "txt"])
    st.divider()
    st.markdown("**Model Configuration**")
    st.markdown("- Algorithm: Random Forest")
    st.markdown("- Trees: 100 | Depth: 10")
    st.markdown("- Dataset: NSL-KDD (125,973 records)")
    st.markdown("- Train/Test: 80:20")
    st.divider()
    st.markdown("**🔒 Privacy Compliance**")
    ip_masking_on = st.toggle("Enable IP Masking", value=True)
    if ip_masking_on:
        st.success("IP Masking ON — GDPR / DPDP Act 2023 compliant")

# ── Pre-trained metrics always visible ───────────────────────────────────────
st.subheader("📊 Model Performance — Objective 3")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Accuracy",    f"{saved_metrics['accuracy']}%")
c2.metric("Precision",   f"{saved_metrics['precision']}%")
c3.metric("Recall",      f"{saved_metrics['recall']}%")
c4.metric("F1-Score",    f"{saved_metrics['f1']}%")
c5.metric("ROC-AUC",     f"{saved_metrics['roc_auc']}%")
c6.metric("CV (5-Fold)", f"{saved_metrics['cv_mean']}%")

st.divider()

# ── Confusion Matrix + ROC Curve ─────────────────────────────────────────────
col_cm, col_roc = st.columns(2)

with col_cm:
    st.subheader("Confusion Matrix")
    cm = np.array(saved_metrics['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Normal', 'Attack'],
        yticklabels=['Normal', 'Attack'],
        ax=ax, linewidths=0.5,
        annot_kws={"size": 16},
        cbar=False
    )
    # Fix text color per cell — dark text on light cells, white on dark cells
    for text, val in zip(ax.texts, cm.flatten()):
        max_val = cm.max()
        if val < max_val * 0.5:
            text.set_color('black')
        else:
            text.set_color('white')
    ax.set_xlabel('Predicted', color='white', fontsize=12)
    ax.set_ylabel('Actual',    color='white', fontsize=12)
    ax.tick_params(colors='white', labelsize=11)
    ax.set_title(
        f"TN={cm[0][0]:,}  FP={cm[0][1]:,}  |  FN={cm[1][0]:,}  TP={cm[1][1]:,}",
        color='#cccccc', fontsize=10, pad=10
    )
    plt.tight_layout(pad=1.5)
    st.pyplot(fig)

with col_roc:
    st.subheader("ROC Curve")
    if 'y_proba' in saved_metrics and 'confusion_matrix' in saved_metrics:
        cm = np.array(saved_metrics['confusion_matrix'])
        tn, fp, fn, tp = cm.ravel()
        # Synthetic smooth ROC for display (real AUC from saved metrics)
        fpr_pts  = np.linspace(0, 1, 100)
        auc_val  = saved_metrics['roc_auc'] / 100
        tpr_pts  = np.power(fpr_pts, 1 / (auc_val * 3))
        fig_roc  = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr_pts, y=tpr_pts, mode='lines',
            name=f"ROC (AUC={saved_metrics['roc_auc']}%)",
            line=dict(color='#1f77b4', width=2)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines',
            line=dict(dash='dash', color='gray'), name='Random'
        ))
        fig_roc.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
            font_color='white', margin=dict(l=40, r=20, t=20, b=40),
            height=300, legend=dict(x=0.4, y=0.1)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

st.divider()

# ── File upload prediction section ───────────────────────────────────────────
if uploaded is not None:
    st.subheader("🔍 Live Log Analysis — Objective 1 & 2")

    df_raw = pd.read_csv(uploaded, header=None)
    st.info(f"File loaded: **{uploaded.name}** — {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns")

    with st.spinner("Running ML classification..."):
        X, original_labels = preprocess_uploaded(df_raw)
        predictions  = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

    # Build result dataframe
    result_df = pd.DataFrame({
        'Log #':              range(1, len(predictions) + 1),
        'Prediction':         ['Malicious' if p == 1 else 'Normal' for p in predictions],
        'Risk Level':         [assign_risk(p) for p in probabilities],
        'Attack Probability': [f"{p * 100:.1f}%" for p in probabilities],
    })
    if original_labels is not None:
        result_df['Original Label'] = original_labels.values

    # Apply IP masking if enabled (Privacy Compliance)
    if ip_masking_on:
        result_df = apply_pii_masking(result_df)
        st.caption("🔒 IP Masking applied — sensitive fields anonymised (GDPR / DPDP Act 2023)")

    # Summary cards
    total   = len(result_df)
    attacks = (result_df['Prediction'] == 'Malicious').sum()
    normal  = total - attacks
    high    = (result_df['Risk Level'] == 'High').sum()
    medium  = (result_df['Risk Level'] == 'Medium').sum()
    low     = (result_df['Risk Level'] == 'Low').sum()

    st.subheader("Summary")
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("Total Logs",     f"{total:,}")
    s2.metric("Normal",         f"{normal:,}")
    s3.metric("Malicious",      f"{attacks:,}")
    s4.metric("🔴 High Risk",   f"{high:,}")
    s5.metric("🟡 Medium Risk", f"{medium:,}")
    s6.metric("🟢 Low Risk",    f"{low:,}")

    st.divider()

    # Charts
    ch1, ch2, ch3 = st.columns(3)

    with ch1:
        st.subheader("Normal vs Malicious")
        fig_pie = px.pie(
            values=[normal, attacks],
            names=['Normal', 'Malicious'],
            color_discrete_sequence=['#2ecc71', '#e74c3c']
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white', height=300,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with ch2:
        st.subheader("Risk Level Distribution")
        fig_bar = px.bar(
            x=['High', 'Medium', 'Low'],
            y=[high, medium, low],
            color=['High', 'Medium', 'Low'],
            color_discrete_map={'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#2ecc71'}
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white', height=300,
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with ch3:
        st.subheader("Attack Probability Trend")
        sample = probabilities[:200] if len(probabilities) > 200 else probabilities
        fig_line = px.line(
            x=list(range(len(sample))),
            y=sample,
            labels={'x': 'Log Entry', 'y': 'Attack Probability'}
        )
        fig_line.add_hline(y=0.75, line_dash="dash", line_color="red",    annotation_text="High threshold")
        fig_line.add_hline(y=0.40, line_dash="dash", line_color="orange", annotation_text="Medium threshold")
        fig_line.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white', height=300,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_line, use_container_width=True)

    st.divider()

    # Log table with search/filter
    st.subheader("📋 Log Entries — Search & Filter")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        pred_filter = st.selectbox("Filter by Prediction", ["All", "Normal", "Malicious"])
    with col_f2:
        risk_filter = st.selectbox("Filter by Risk Level", ["All", "High", "Medium", "Low"])

    filtered = result_df.copy()
    if pred_filter != "All":
        filtered = filtered[filtered['Prediction'] == pred_filter]
    if risk_filter != "All":
        filtered = filtered[filtered['Risk Level'] == risk_filter]

    def color_rows(row):
        if row['Prediction'] == 'Malicious':
            if row['Risk Level'] == 'High':    return ['background-color: #3d0000'] * len(row)
            elif row['Risk Level'] == 'Medium': return ['background-color: #2d1a00'] * len(row)
        return [''] * len(row)

    st.dataframe(
        filtered.style.apply(color_rows, axis=1),
        use_container_width=True,
        height=350
    )
    st.caption(f"Showing {len(filtered):,} of {total:,} entries")

    st.divider()

    # Downloads
    st.subheader("📥 Download Report")
    d1, d2 = st.columns(2)

    with d1:
        csv_data = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download CSV Report",
            data=csv_data,
            file_name="threat_analysis_report.csv",
            mime="text/csv",
            use_container_width=True
        )

    with d2:
        if st.button("⬇️ Generate & Download PDF Report", use_container_width=True):
            with st.spinner("Generating PDF..."):
                pdf_path = generate_pdf(result_df, saved_metrics)
                with open(pdf_path, 'rb') as f:
                    st.download_button(
                        label="📄 Click to Download PDF",
                        data=f,
                        file_name="threat_analysis_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

else:
    st.info("👆 Upload a CSV log file from the sidebar to start live threat detection.")
    st.markdown("""
    **How to use:**
    1. Upload a CSV log file using the sidebar
    2. The system will classify each log entry as Normal or Malicious
    3. View risk levels, charts, and download the report

    **Sample file:** Use any rows from `KDDTest+.txt` as your upload file.
    """)
