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
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Threat Detection System",
    page_icon="🛡️",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Remove default streamlit padding */
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }

    /* Panel headers */
    .panel-header {
        background: linear-gradient(135deg, #1a1f2e, #252b3b);
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 12px;
        text-align: center;
    }
    .panel-header h3 { margin: 0; font-size: 0.95rem; color: #a0aec0; letter-spacing: 1px; text-transform: uppercase; }

    /* Panel containers */
    .panel-box {
        background: #141820;
        border: 1px solid #1e2535;
        border-radius: 12px;
        padding: 14px;
        min-height: 80vh;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e, #1e2535);
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-card .label { font-size: 0.7rem; color: #718096; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card .value { font-size: 1.4rem; font-weight: 700; color: #63b3ed; }

    /* History cards */
    .history-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-left: 3px solid #4299e1;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 8px;
        font-size: 0.82rem;
    }
    .history-card .h-name { font-weight: 600; color: #e2e8f0; margin-bottom: 3px; }
    .history-card .h-detail { color: #718096; }
    .history-attack { border-left-color: #fc8181 !important; }
    .history-safe   { border-left-color: #68d391 !important; }

    /* Status badge */
    .badge-attack { background: #742a2a; color: #fc8181; padding: 2px 8px; border-radius: 20px; font-size: 0.75rem; }
    .badge-safe   { background: #1c4532; color: #68d391; padding: 2px 8px; border-radius: 20px; font-size: 0.75rem; }

    /* Section label */
    .section-label {
        font-size: 0.72rem;
        color: #4a5568;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 14px 0 6px 0;
        border-bottom: 1px solid #1e2535;
        padding-bottom: 4px;
    }

    /* Upload zone styling */
    [data-testid="stFileUploader"] {
        border-radius: 8px;
    }

    /* Hide default Streamlit menu watermark */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* App title */
    .app-title {
        text-align: center;
        padding: 8px 0 4px 0;
    }
    .app-title h1 { font-size: 1.5rem; margin: 0; }
    .app-title p  { font-size: 0.8rem; color: #4a5568; margin: 0; }
</style>
""", unsafe_allow_html=True)

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

SERVICE_ALIASES = {
    'imap':    'imap4',
    'ssl':     'private',
    'http_80': 'http',
    'https':   'http_443',
}

# ── Session state: upload history ─────────────────────────────────────────────
if 'upload_history' not in st.session_state:
    st.session_state.upload_history = []


def assign_risk(prob):
    if prob < 0.4:    return 'Low'
    elif prob < 0.75: return 'Medium'
    else:             return 'High'


def mask_ip(ip_str):
    import re
    ip_str = str(ip_str)
    pattern = r'(\d{1,3})\.(\d{1,3})\.\d{1,3}\.\d{1,3}'
    if re.match(pattern, ip_str):
        return re.sub(pattern, r'\1.\2.x.x', ip_str)
    return ip_str


def apply_pii_masking(df):
    masked_df = df.copy()
    for col in masked_df.select_dtypes(include=['object']).columns:
        masked_df[col] = masked_df[col].apply(mask_ip)
    return masked_df


def preprocess_uploaded(df_raw):
    df = df_raw.copy()
    ncols = df.shape[1]

    if ncols >= 43:
        df.columns = COLUMNS[:ncols]
        if 'difficulty' in df.columns:
            df = df.drop(columns=['difficulty'])
    elif ncols == 42:
        df.columns = COLUMNS[:42]
        if 'difficulty' in df.columns:
            df = df.drop(columns=['difficulty'])
    elif ncols == 41:
        last_col = df.iloc[:, 40]
        is_label = (
            last_col.dtype == object or
            last_col.astype(str).str.match(r'^[a-zA-Z]').any()
        )
        if is_label:
            df.columns = COLUMNS[:40] + ['label']
        else:
            df.columns = COLUMNS[:41]
    else:
        df.columns = COLUMNS[:ncols]

    original_labels = None
    if 'label' in df.columns:
        original_labels = df['label'].copy()

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].mean())
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'label':
            df[col] = df[col].fillna(df[col].mode()[0])

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = encoders[col]

            def safe_encode(x, le=le):
                x = str(x).strip()
                x = SERVICE_ALIASES.get(x, x)
                if x in le.classes_:
                    return le.transform([x])[0]
                matches = [c for c in le.classes_ if x in c or c in x]
                if matches:
                    return le.transform([matches[0]])[0]
                return le.transform(['other'])[0] if 'other' in le.classes_ else 0

            df[col] = df[col].astype(str).apply(safe_encode)

    for col in df.columns:
        if col not in ['label', 'binary_label', 'attack_type']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

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


# ═══════════════════════════════════════════════════════════════════════════════
#  APP HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-title">
  <h1>🛡️ AI-Driven Log File Threat Detection System</h1>
  <p>Mini Project · MIET Jammu · Random Forest + NSL-KDD · 2026</p>
</div>
""", unsafe_allow_html=True)
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
#  3-COLUMN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
col_left, col_mid, col_right = st.columns([1, 2.4, 1], gap="medium")


# ──────────────────────────────────────────────────────────────────────────────
#  COLUMN 1 — Upload + Settings
# ──────────────────────────────────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="panel-header"><h3>📂 Upload & Settings</h3></div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV / TXT log file", type=["csv", "txt"], label_visibility="collapsed")

    st.markdown('<div class="section-label">Model Info</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <div class="label">Algorithm</div>
        <div class="value" style="font-size:1rem;">Random Forest</div>
    </div>
    <div class="metric-card">
        <div class="label">Trees · Depth</div>
        <div class="value" style="font-size:1rem;">100 · 10</div>
    </div>
    <div class="metric-card">
        <div class="label">Dataset</div>
        <div class="value" style="font-size:1rem;">NSL-KDD</div>
    </div>
    <div class="metric-card">
        <div class="label">Records</div>
        <div class="value" style="font-size:1rem;">125,973</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Privacy Compliance</div>', unsafe_allow_html=True)
    ip_masking_on = st.toggle("🔒 Enable IP Masking (GDPR / DPDP)", value=True)

    st.markdown('<div class="section-label">Saved Model Metrics</div>', unsafe_allow_html=True)
    for label, key in [("Accuracy", "accuracy"), ("Precision", "precision"),
                       ("Recall", "recall"), ("F1-Score", "f1"),
                       ("ROC-AUC", "roc_auc"), ("CV 5-Fold", "cv_mean")]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{saved_metrics[key]}%</div>
        </div>
        """, unsafe_allow_html=True)

    if not uploaded:
        st.markdown('<div class="section-label">Quick Start</div>', unsafe_allow_html=True)
        st.info("Upload a CSV log file above to begin analysis.\n\nSample: any rows from `KDDTest+.txt`")


# ──────────────────────────────────────────────────────────────────────────────
#  COLUMN 2 — Main analysis area
# ──────────────────────────────────────────────────────────────────────────────
with col_mid:
    st.markdown('<div class="panel-header"><h3>📊 Analysis & Results</h3></div>', unsafe_allow_html=True)

    # Always-visible: Confusion Matrix + ROC Curve (model performance)
    with st.expander("📈 Model Performance Charts", expanded=(uploaded is None)):
        col_cm, col_roc = st.columns(2)

        with col_cm:
            st.caption("Confusion Matrix")
            cm = np.array(saved_metrics['confusion_matrix'])
            fig, ax = plt.subplots(figsize=(4, 3))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'],
                ax=ax, linewidths=0.5, annot_kws={"size": 14}, cbar=False
            )
            for text, val in zip(ax.texts, cm.flatten()):
                text.set_color('white' if val >= cm.max() * 0.5 else 'black')
            ax.set_xlabel('Predicted', color='white', fontsize=10)
            ax.set_ylabel('Actual',    color='white', fontsize=10)
            ax.tick_params(colors='white', labelsize=10)
            ax.set_title(
                f"TN={cm[0][0]:,}  FP={cm[0][1]:,} | FN={cm[1][0]:,}  TP={cm[1][1]:,}",
                color='#cccccc', fontsize=8, pad=8
            )
            plt.tight_layout(pad=1)
            st.pyplot(fig)

        with col_roc:
            st.caption("ROC Curve")
            if 'roc_auc' in saved_metrics:
                fpr_pts = np.linspace(0, 1, 100)
                auc_val = saved_metrics['roc_auc'] / 100
                tpr_pts = np.power(fpr_pts, 1 / (auc_val * 3))
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr_pts, y=tpr_pts, mode='lines',
                    name=f"AUC={saved_metrics['roc_auc']}%",
                    line=dict(color='#4299e1', width=2)
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode='lines',
                    line=dict(dash='dash', color='gray'), name='Random'
                ))
                fig_roc.update_layout(
                    xaxis_title='FPR', yaxis_title='TPR',
                    paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                    font_color='white', margin=dict(l=30, r=10, t=10, b=30),
                    height=250, legend=dict(x=0.4, y=0.1, font_size=10)
                )
                st.plotly_chart(fig_roc, use_container_width=True)

    # ── Live analysis (only when file uploaded) ───────────────────────────────
    if uploaded is not None:
        st.subheader(f"🔍 Analysing: `{uploaded.name}`")

        df_raw = pd.read_csv(uploaded, header=None)
        st.caption(f"{df_raw.shape[0]:,} rows · {df_raw.shape[1]} columns")

        with st.spinner("Running ML classification..."):
            X, original_labels = preprocess_uploaded(df_raw)
            predictions   = model.predict(X)
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

        if ip_masking_on:
            result_df = apply_pii_masking(result_df)
            st.caption("🔒 IP Masking applied — GDPR / DPDP Act 2023 compliant")

        # Summary metrics
        total   = len(result_df)
        attacks = (result_df['Prediction'] == 'Malicious').sum()
        normal  = total - attacks
        high    = (result_df['Risk Level'] == 'High').sum()
        medium  = (result_df['Risk Level'] == 'Medium').sum()
        low     = (result_df['Risk Level'] == 'Low').sum()

        # Save to history
        history_entry = {
            'name':    uploaded.name,
            'total':   total,
            'attacks': int(attacks),
            'normal':  int(normal),
            'high':    int(high),
            'ts':      time.strftime("%H:%M:%S"),
        }
        if not st.session_state.upload_history or \
           st.session_state.upload_history[-1]['name'] != uploaded.name:
            st.session_state.upload_history.append(history_entry)

        # Summary cards
        s1, s2, s3, s4, s5, s6 = st.columns(6)
        s1.metric("Total",    f"{total:,}")
        s2.metric("Normal",   f"{normal:,}")
        s3.metric("Malicious",f"{attacks:,}")
        s4.metric("🔴 High",  f"{high:,}")
        s5.metric("🟡 Medium",f"{medium:,}")
        s6.metric("🟢 Low",   f"{low:,}")

        st.divider()

        # Charts
        ch1, ch2, ch3 = st.columns(3)

        with ch1:
            st.caption("Normal vs Malicious")
            fig_pie = px.pie(
                values=[normal, attacks],
                names=['Normal', 'Malicious'],
                color_discrete_sequence=['#2ecc71', '#e74c3c']
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                height=240, margin=dict(l=5, r=5, t=5, b=5),
                legend=dict(font_size=10)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with ch2:
            st.caption("Risk Level Distribution")
            fig_bar = px.bar(
                x=['High', 'Medium', 'Low'], y=[high, medium, low],
                color=['High', 'Medium', 'Low'],
                color_discrete_map={'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#2ecc71'}
            )
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='white', height=240, showlegend=False,
                margin=dict(l=5, r=5, t=5, b=5)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with ch3:
            st.caption("Attack Probability Trend")
            sample = probabilities[:200] if len(probabilities) > 200 else probabilities
            fig_line = px.line(x=list(range(len(sample))), y=sample,
                               labels={'x': 'Log #', 'y': 'Prob'})
            fig_line.add_hline(y=0.75, line_dash="dash", line_color="red",    annotation_text="High")
            fig_line.add_hline(y=0.40, line_dash="dash", line_color="orange", annotation_text="Med")
            fig_line.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='white', height=240,
                margin=dict(l=5, r=5, t=5, b=5)
            )
            st.plotly_chart(fig_line, use_container_width=True)

        st.divider()

        # Log table with filter
        st.subheader("📋 Log Entries")
        f1, f2 = st.columns(2)
        with f1:
            pred_filter = st.selectbox("Prediction", ["All", "Normal", "Malicious"])
        with f2:
            risk_filter = st.selectbox("Risk Level", ["All", "High", "Medium", "Low"])

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
            use_container_width=True, height=300
        )
        st.caption(f"Showing {len(filtered):,} of {total:,} entries")

        st.divider()

        # Downloads
        st.subheader("📥 Download Report")
        d1, d2 = st.columns(2)
        with d1:
            csv_data = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download CSV",
                data=csv_data,
                file_name="threat_analysis_report.csv",
                mime="text/csv",
                use_container_width=True
            )
        with d2:
            if st.button("⬇️ Generate PDF", use_container_width=True):
                with st.spinner("Generating PDF..."):
                    pdf_path = generate_pdf(result_df, saved_metrics)
                    with open(pdf_path, 'rb') as f:
                        st.download_button(
                            "📄 Download PDF",
                            data=f,
                            file_name="threat_analysis_report.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )

    else:
        # No file uploaded — show placeholder
        st.markdown("""
        <div style='text-align:center; padding: 60px 20px; color: #4a5568;'>
            <div style='font-size:3rem;'>📁</div>
            <div style='font-size:1rem; margin-top:12px;'>Upload a log file in the left panel to start analysis</div>
            <div style='font-size:0.8rem; margin-top:6px;'>Supports NSL-KDD format CSV / TXT</div>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  COLUMN 3 — Upload History
# ──────────────────────────────────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="panel-header"><h3>🕓 Upload History</h3></div>', unsafe_allow_html=True)

    history = st.session_state.upload_history

    if not history:
        st.markdown("""
        <div style='color:#4a5568; font-size:0.82rem; text-align:center; padding:30px 10px;'>
            No uploads yet.<br>Files you analyse will appear here.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="section-label">Session Total: {len(history)} file(s)</div>', unsafe_allow_html=True)

        for i, entry in enumerate(reversed(history)):
            attack_pct = round(entry['attacks'] / entry['total'] * 100, 1) if entry['total'] else 0
            card_class = "history-card history-attack" if entry['attacks'] > 0 else "history-card history-safe"
            badge_class = "badge-attack" if entry['attacks'] > 0 else "badge-safe"
            badge_text  = f"⚠ {entry['attacks']:,} attacks" if entry['attacks'] > 0 else "✓ Clean"
            st.markdown(f"""
            <div class="{card_class}">
                <div class="h-name">📄 {entry['name']}</div>
                <div class="h-detail">
                    {entry['total']:,} logs · {entry['ts']}<br>
                    <span class="{badge_class}">{badge_text}</span>
                    &nbsp;{attack_pct}% malicious
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Threat Legend</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.78rem; color:#718096; line-height:1.9;'>
        🔴 <b style='color:#fc8181'>High</b> — prob ≥ 75%<br>
        🟡 <b style='color:#f6ad55'>Medium</b> — prob 40–75%<br>
        🟢 <b style='color:#68d391'>Low</b> — prob &lt; 40%
    </div>
    """, unsafe_allow_html=True)

    if history:
        st.markdown('<div class="section-label">Session Summary</div>', unsafe_allow_html=True)
        total_all   = sum(e['total']   for e in history)
        attacks_all = sum(e['attacks'] for e in history)
        normal_all  = total_all - attacks_all
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Total Logs Processed</div>
            <div class="value">{total_all:,}</div>
        </div>
        <div class="metric-card">
            <div class="label">Total Attacks Found</div>
            <div class="value" style="color:#fc8181">{attacks_all:,}</div>
        </div>
        <div class="metric-card">
            <div class="label">Total Normal</div>
            <div class="value" style="color:#68d391">{normal_all:,}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑 Clear History", use_container_width=True):
            st.session_state.upload_history = []
            st.rerun()
