import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.units import inch

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="SleepWise", page_icon="🌙", layout="centered")

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background-color: #0f1117; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; }
.title-block { text-align: center; padding: 2rem 0 1rem 0; }
.title-block h1 { font-size: 3rem; color: #e2e8f0; margin-bottom: 0.2rem; }
.title-block p { color: #94a3b8; font-size: 1.05rem; font-weight: 300; }
.section-header { color: #7c9cbf; font-size: 0.75rem; font-weight: 500; text-transform: uppercase;
    letter-spacing: 0.12em; margin: 2rem 0 0.8rem 0; border-bottom: 1px solid #1e293b; padding-bottom: 0.4rem; }
.result-box { border-radius: 16px; padding: 2rem; text-align: center; margin: 2rem 0; }
.result-good     { background: linear-gradient(135deg, #064e3b, #065f46); border: 1px solid #10b981; }
.result-moderate { background: linear-gradient(135deg, #1e3a5f, #1e40af); border: 1px solid #3b82f6; }
.result-poor     { background: linear-gradient(135deg, #4c1d1d, #7f1d1d); border: 1px solid #ef4444; }
.result-box h2 { font-size: 2rem; margin-bottom: 0.3rem; color: #f1f5f9; }
.result-box p  { color: #cbd5e1; font-size: 1rem; }
.score-box { background: #1e293b; border-radius: 16px; padding: 1.5rem; text-align: center; margin: 1rem 0; }
.score-number { font-size: 3.5rem; font-family: 'DM Serif Display', serif; line-height: 1; margin-bottom: 0.3rem; }
.score-label  { color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; }
.rec-card { background: #1e293b; border-radius: 12px; padding: 1rem 1.2rem; margin: 0.5rem 0;
    border-left: 3px solid #3b82f6; color: #e2e8f0; font-size: 0.95rem; }
.whatif-box { background: #1e293b; border-radius: 16px; padding: 1.5rem; margin: 1rem 0; border: 1px solid #334155; }
.stButton > button { width: 100%; background: linear-gradient(135deg, #1d4ed8, #3b82f6); color: white;
    border: none; border-radius: 10px; padding: 0.8rem; font-size: 1rem; font-family: 'DM Sans', sans-serif;
    font-weight: 500; margin-top: 1.5rem; cursor: pointer; }
div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label { color: #94a3b8 !important; font-size: 0.9rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Normalize prediction string ──────────────────────────────
def normalize_prediction(pred):
    pred = str(pred).strip().lower()
    if "good"     in pred: return "Good Sleep"
    if "moderate" in pred: return "Moderate Sleep"
    if "poor"     in pred: return "Poor Sleep"
    return pred

# ── Prediction style lookup ──────────────────────────────────
def get_style(prediction):
    if prediction == "Good Sleep":
        return "result-good",     "✨", "Your lifestyle supports healthy, restorative sleep.",    "#10b981"
    elif prediction == "Moderate Sleep":
        return "result-moderate", "🌤", "Your sleep is decent but has room for improvement.",     "#3b82f6"
    else:
        return "result-poor",     "⚠️", "Your lifestyle may be disrupting your sleep quality.", "#ef4444"

# ── Compute sleep score ──────────────────────────────────────
def compute_score(sleep_duration, awakenings, caffeine, alcohol, smoking, exercise):
    score = 0
    # Sleep duration - max 35 points
    if 7 <= sleep_duration <= 9:                               score += 35
    elif 6 <= sleep_duration < 7 or 9 < sleep_duration <= 10: score += 23
    else:                                                       score += 8
    # Awakenings - max 25 points
    score += max(0, 25 - awakenings * 6)
    # Lifestyle - max 40 points
    if caffeine == 0:    score += 8
    elif caffeine <= 50: score += 5
    elif caffeine <= 100:score += 2
    if alcohol == 0:     score += 8
    elif alcohol <= 2:   score += 5
    elif alcohol <= 4:   score += 2
    if smoking == "No":  score += 8
    if exercise >= 4:    score += 10
    elif exercise >= 3:  score += 8
    elif exercise >= 1:  score += 4
    return int(min(score, 100))

# ── Score bar color matches prediction ──────────────────────
def get_bar_color(prediction):
    if prediction == "Good Sleep":     return "#10b981"
    if prediction == "Moderate Sleep": return "#3b82f6"
    return "#ef4444"

# ── Make prediction ──────────────────────────────────────────
def make_prediction(model, scaler, age, gender, bedtime_hour, wakeup_hour,
                    sleep_duration, rem, deep, light, awakenings,
                    caffeine, alcohol, smoking, exercise):
    gender_encoded  = 0 if gender == "Female" else 1
    smoking_encoded = 1 if smoking == "No"    else 0
    input_data = pd.DataFrame([{
        "ID": 1, "Age": age, "Gender": gender_encoded,
        "Bedtime": bedtime_hour, "Wakeup time": wakeup_hour,
        "Sleep duration": sleep_duration,
        "REM sleep percentage": rem,
        "Deep sleep percentage": deep,
        "Light sleep percentage": light,
        "Awakenings": float(awakenings),
        "Caffeine consumption": float(caffeine),
        "Alcohol consumption": float(alcohol),
        "Smoking status": smoking_encoded,
        "Exercise frequency": float(exercise),
    }])
    raw = model.predict(scaler.transform(input_data))[0]
    return normalize_prediction(raw)

# ── Generate PDF ─────────────────────────────────────────────
def generate_pdf(prediction, score, orig, recs, insights):
    buffer = BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=A4,
                               rightMargin=0.75*inch, leftMargin=0.75*inch,
                               topMargin=0.75*inch,  bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story  = []

    title_s = ParagraphStyle("t", parent=styles["Title"],   fontSize=22, textColor=colors.HexColor("#1d4ed8"), spaceAfter=4)
    sub_s   = ParagraphStyle("s", parent=styles["Normal"],  fontSize=11, textColor=colors.HexColor("#64748b"), spaceAfter=16)
    head_s  = ParagraphStyle("h", parent=styles["Heading2"],fontSize=13, textColor=colors.HexColor("#1e293b"), spaceBefore=14, spaceAfter=6)
    body_s  = ParagraphStyle("b", parent=styles["Normal"],  fontSize=10, textColor=colors.HexColor("#334155"), spaceAfter=6)
    foot_s  = ParagraphStyle("f", parent=styles["Normal"],  fontSize=8,  textColor=colors.HexColor("#94a3b8"))

    story.append(Paragraph("SleepWise", title_s))
    story.append(Paragraph("Personalised Sleep Quality Report", sub_s))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 12))

    pred_color = "#10b981" if prediction=="Good Sleep" else "#3b82f6" if prediction=="Moderate Sleep" else "#ef4444"
    story.append(Paragraph(f'<font color="{pred_color}"><b>Prediction: {prediction}</b></font>', head_s))
    story.append(Paragraph(f"Sleep Score: <b>{score} / 100</b>", body_s))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Your Inputs", head_s))
    tbl_data = [["Factor","Your Value"]] + [[k, str(v)] for k,v in orig.items()]
    tbl = Table(tbl_data, colWidths=[3*inch, 2*inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1d4ed8")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTSIZE",(0,0),(-1,0),10), ("FONTSIZE",(0,1),(-1,-1),9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f8fafc"),colors.white]),
        ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#e2e8f0")),
        ("PADDING",(0,0),(-1,-1),6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Comparison to Average Good Sleeper", head_s))
    for i in insights:
        clean = i.replace("<strong>","<b>").replace("</strong>","</b>")
        story.append(Paragraph(f"• {clean}", body_s))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Personalised Recommendations", head_s))
    for r in recs:
        story.append(Paragraph(f"• {r}", body_s))
    story.append(Spacer(1, 20))

    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Generated by SleepWise — Group D-11 | AI Semester Project", foot_s))
    doc.build(story)
    buffer.seek(0)
    return buffer

# ── Load artifacts ───────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("sleepwise_model.pkl",  "rb") as f: model   = pickle.load(f)
    with open("sleepwise_scaler.pkl", "rb") as f: scaler  = pickle.load(f)
    with open("sleepwise_encoder.pkl","rb") as f: encoder = pickle.load(f)
    return model, scaler, encoder

model, scaler, encoder = load_artifacts()

if "history" not in st.session_state:
    st.session_state["history"] = []

# ── Navigation ───────────────────────────────────────────────
page = st.sidebar.selectbox("Navigate", ["Predict", "History", "About"])

# ══════════════════════════════════════════════════════════════
# ABOUT PAGE
# ══════════════════════════════════════════════════════════════
if page == "About":
    st.markdown('<div class="title-block"><h1>🌙 SleepWise</h1><p>AI-powered sleep quality prediction based on your lifestyle</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">What is SleepWise?</div>', unsafe_allow_html=True)
    st.markdown('<div class="rec-card" style="border-left-color:#10b981;">SleepWise is an AI-powered application that predicts your sleep quality based on your lifestyle and sleeping habits. By analyzing factors like sleep duration, exercise, caffeine intake, and more — SleepWise gives you a personalized sleep quality prediction and actionable recommendations to help you sleep better.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">How Does It Work?</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    for col,icon,step,desc in [
        (c1,"📋","Step 1","Enter your sleep patterns and lifestyle information"),
        (c2,"🤖","Step 2","Our AI model analyzes your inputs using trained ML algorithms"),
        (c3,"✨","Step 3","Get your sleep quality prediction and personalized recommendations"),
    ]:
        with col:
            st.markdown(f'<div class="score-box"><div style="font-size:2rem;">{icon}</div><div style="color:#e2e8f0;font-weight:500;margin:0.5rem 0;">{step}</div><div style="color:#94a3b8;font-size:0.85rem;">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">The AI Model</div>', unsafe_allow_html=True)
    for txt in [
        "🧠 <strong>Algorithm:</strong> Support Vector Machine (SVM) — best performing after comparing Logistic Regression, Decision Tree, Random Forest, and SVM.",
        "📊 <strong>Accuracy:</strong> 76% on the test dataset with a weighted F1-Score of 0.76.",
        "🗂️ <strong>Dataset:</strong> Trained on the Kaggle Sleep Efficiency Dataset with 452 real sleep records.",
        "⚖️ <strong>Classes:</strong> Good Sleep, Moderate Sleep, and Poor Sleep — determined using data-driven tertile boundaries.",
    ]:
        st.markdown(f'<div class="rec-card" style="border-left-color:#8b5cf6;">{txt}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">What Do The Inputs Mean?</div>', unsafe_allow_html=True)
    for label, desc in {
        "🛌 Sleep Duration":          "Total hours of sleep per night. Adults need 7–9 hours for optimal health.",
        "😴 Awakenings":              "Number of times you wake up during the night. Fewer is better.",
        "☕ Caffeine":                "Total caffeine per day in mg. A standard cup of coffee is ~95mg.",
        "🍷 Alcohol":                 "Drinks per week. Alcohol reduces sleep quality even in small amounts.",
        "🚬 Smoking":                 "Nicotine is a stimulant that disrupts sleep patterns significantly.",
        "🏃 Exercise":                "Days per week with physical activity. Regular exercise improves sleep depth.",
    }.items():
        st.markdown(f'<div class="rec-card"><strong>{label}</strong><br><span style="color:#94a3b8;">{desc}</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Built By</div>', unsafe_allow_html=True)
    st.markdown('<div class="rec-card" style="border-left-color:#f59e0b;text-align:center;"><div style="color:#e2e8f0;font-size:1rem;">SleepWise — AI Project, Semester Project</div><div style="color:#94a3b8;font-size:0.85rem;margin-top:0.3rem;">Group D-11</div></div>', unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════
# HISTORY PAGE
# ══════════════════════════════════════════════════════════════
if page == "History":
    st.markdown('<div class="title-block"><h1>🌙 SleepWise</h1><p>Your prediction history this session</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Session History</div>', unsafe_allow_html=True)

    if not st.session_state["history"]:
        st.markdown('<div class="rec-card" style="text-align:center;color:#94a3b8;">No predictions yet. Go to the Predict page to get started!</div>', unsafe_allow_html=True)
    else:
        history_df = pd.DataFrame(st.session_state["history"])
        history_df.index = [f"Entry {i+1}" for i in range(len(history_df))]

        def color_prediction(val):
            if val == "Good Sleep":      return "background-color:#064e3b;color:#10b981"
            elif val == "Moderate Sleep": return "background-color:#1e3a5f;color:#3b82f6"
            else:                         return "background-color:#4c1d1d;color:#ef4444"

        st.dataframe(history_df.style.map(color_prediction, subset=["Prediction"]), use_container_width=True)

        if len(st.session_state["history"]) > 1:
            st.markdown('<div class="section-header">Score Trend</div>', unsafe_allow_html=True)
            scores = [h["Score"] for h in st.session_state["history"]]
            fig_h, ax_h = plt.subplots(figsize=(6, 2.5))
            fig_h.patch.set_facecolor('#1e293b'); ax_h.set_facecolor('#1e293b')
            ax_h.plot(range(1, len(scores)+1), scores, color='#3b82f6', marker='o', linewidth=2, markersize=6)
            ax_h.fill_between(range(1, len(scores)+1), scores, alpha=0.15, color='#3b82f6')
            ax_h.set_xlabel("Entry", color='#94a3b8', fontsize=8)
            ax_h.set_ylabel("Score", color='#94a3b8', fontsize=8)
            ax_h.tick_params(colors='#94a3b8', labelsize=8)
            ax_h.spines['bottom'].set_color('#334155'); ax_h.spines['left'].set_color('#334155')
            ax_h.spines['top'].set_visible(False);     ax_h.spines['right'].set_visible(False)
            ax_h.set_ylim(0, 100)
            plt.tight_layout(); st.pyplot(fig_h); plt.close()

        if st.button("Clear History"):
            st.session_state["history"] = []
            st.rerun()
    st.stop()

# ══════════════════════════════════════════════════════════════
# PREDICT PAGE
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="title-block"><h1>🌙 SleepWise</h1><p>AI-powered sleep quality prediction based on your lifestyle</p></div>', unsafe_allow_html=True)

st.markdown('<div class="section-header">Personal Information</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1: age    = st.number_input("Age", min_value=5, max_value=100, value=25)
with c2: gender = st.selectbox("Gender", ["Female", "Male"])

st.markdown('<div class="section-header">Sleep Patterns</div>', unsafe_allow_html=True)
c3, c4 = st.columns(2)
with c3: sleep_duration = st.slider("Sleep Duration (hours)", 4.0, 12.0, 7.0, 0.5)
with c4: awakenings     = st.slider("Awakenings per night",   0,   5,    1)

with st.expander("💡 Don't know your sleep stage percentages? Click here"):
    st.markdown("""
    Sleep happens in 3 stages that cycle throughout the night:

    | Stage | What it is | Healthy range |
    |---|---|---|
    | 🌙 **Light Sleep** | Lightest stage — easy to wake from. Body starts to relax. | 50–60% |
    | 🧠 **Deep Sleep** | Most restorative. Body repairs muscles, boosts immunity. | 13–23% |
    | 💭 **REM Sleep** | When you dream. Critical for memory and mood. | 20–25% |

    **How to find your numbers:**
    - **Fitbit, Apple Watch, Samsung Watch** — check the Sleep section in the app
    - **Oura Ring** — Sleep Analysis tab in the Oura app
    - **No tracker?** Use these typical averages:
        - Light Sleep: **55%**, Deep Sleep: **20%**, REM Sleep: **25%**
    """)

c5, c6, c7 = st.columns(3)
with c5: rem   = st.slider("REM Sleep %",   10, 35, 25)
with c6: deep  = st.slider("Deep Sleep %",  10, 80, 20)
with c7: light = st.slider("Light Sleep %",  5, 70, 55)

c5, c6 = st.columns(2)
with c5: bedtime_hour = st.selectbox("Bedtime (hour, 24h)",      list(range(24)), index=23)
with c6: wakeup_hour  = st.selectbox("Wake-up time (hour, 24h)", list(range(24)), index=7)

st.markdown('<div class="section-header">Lifestyle Factors</div>', unsafe_allow_html=True)
c7, c8 = st.columns(2)
with c7: caffeine = st.selectbox("Caffeine consumption (mg/day)",     [0, 25, 50, 75, 100, 150, 200])
with c8: alcohol  = st.selectbox("Alcohol consumption (drinks/week)", [0, 1, 2, 3, 4, 5])

c9, c10 = st.columns(2)
with c9:  smoking  = st.selectbox("Smoking status",                ["No", "Yes"])
with c10: exercise = st.selectbox("Exercise frequency (days/week)", [0, 1, 2, 3, 4, 5])

# ── Predict button ───────────────────────────────────────────
if st.button("Predict My Sleep Quality"):
    prediction = make_prediction(model, scaler, age, gender, bedtime_hour, wakeup_hour,
                                 sleep_duration, rem, deep, light, awakenings,
                                 caffeine, alcohol, smoking, exercise)
    score = compute_score(sleep_duration, awakenings, caffeine, alcohol, smoking, exercise)

    st.session_state.update({
        "predicted": True, "prediction": prediction, "score": score,
        "orig_age": age, "orig_gender": gender,
        "orig_bedtime": bedtime_hour, "orig_wakeup": wakeup_hour,
        "orig_sleep": sleep_duration, "orig_awakenings": awakenings,
        "orig_caffeine": caffeine, "orig_alcohol": alcohol,
        "orig_smoking": smoking, "orig_exercise": exercise,
    })
    st.session_state["history"].append({
        "Prediction": prediction, "Score": score,
        "Sleep (hrs)": sleep_duration, "Awakenings": awakenings,
        "Caffeine": caffeine, "Alcohol": alcohol,
        "Exercise": exercise, "Smoking": smoking,
    })

# ── Show results ──────────────────────────────────────────────
if st.session_state.get("predicted"):

    prediction  = st.session_state["prediction"]
    score       = st.session_state["score"]
    orig_sleep  = st.session_state["orig_sleep"]
    orig_awk    = st.session_state["orig_awakenings"]
    orig_caf    = st.session_state["orig_caffeine"]
    orig_alc    = st.session_state["orig_alcohol"]
    orig_smk    = st.session_state["orig_smoking"]
    orig_exe    = st.session_state["orig_exercise"]
    orig_age    = st.session_state["orig_age"]
    orig_gender = st.session_state["orig_gender"]

    css_class, emoji, message, score_color = get_style(prediction)
    bar_color = get_bar_color(prediction)   # ← score bar always matches prediction

    # ── Result box ───────────────────────────────────────────
    st.markdown(f'<div class="result-box {css_class}"><h2>{emoji} {prediction}</h2><p>{message}</p></div>', unsafe_allow_html=True)

    # ── Sleep Score ──────────────────────────────────────────
    st.markdown('<div class="section-header">Sleep Score</div>', unsafe_allow_html=True)
    cs, cg = st.columns([1, 2])
    with cs:
        st.markdown(f'<div class="score-box"><div class="score-number" style="color:{score_color};">{score}</div><div class="score-label">out of 100</div></div>', unsafe_allow_html=True)
    with cg:
        st.markdown(f"""
        <div style="background:#1e293b;border-radius:12px;padding:1.2rem;margin-top:0.5rem;">
            <div style="color:#94a3b8;font-size:0.8rem;margin-bottom:0.5rem;">What your score means</div>
            <div style="color:{bar_color};font-size:0.95rem;font-weight:500;">{emoji} {prediction}</div>
            <div style="color:#94a3b8;font-size:0.85rem;margin-top:0.4rem;">{message}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Factor breakdown ─────────────────────────────────────
    st.markdown('<div class="section-header">How Your Factors Compare</div>', unsafe_allow_html=True)
    factors = {
        "Sleep Duration": min(orig_sleep / 9, 1),
        "Low Awakenings": max(0, 1 - orig_awk / 5),
        "No Caffeine":    1 - orig_caf / 200,
        "No Alcohol":     1 - orig_alc / 5,
        "Exercise":       orig_exe / 5,
        "Non-Smoker":     0 if orig_smk == "Yes" else 1,
    }
    flabels = list(factors.keys())
    fvalues = [round(v * 100) for v in factors.values()]
    fcolors = ["#10b981" if v >= 70 else "#f59e0b" if v >= 40 else "#ef4444" for v in fvalues]
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    fig2.patch.set_facecolor('#1e293b'); ax2.set_facecolor('#1e293b')
    bars = ax2.barh(flabels, fvalues, color=fcolors, edgecolor='none', height=0.55)
    for bar, val in zip(bars, fvalues):
        ax2.text(bar.get_width()+1.5, bar.get_y()+bar.get_height()/2, f'{val}%', va='center', ha='left', color='#94a3b8', fontsize=8)
    ax2.set_xlim(0, 115); ax2.set_xlabel("Score (%)", color='#94a3b8', fontsize=8)
    ax2.tick_params(colors='#94a3b8', labelsize=8)
    ax2.spines['bottom'].set_color('#334155'); ax2.spines['left'].set_color('#334155')
    ax2.spines['top'].set_visible(False);     ax2.spines['right'].set_visible(False)
    ax2.legend(handles=[
        mpatches.Patch(color='#10b981', label='Good (≥70%)'),
        mpatches.Patch(color='#f59e0b', label='Fair (40-69%)'),
        mpatches.Patch(color='#ef4444', label='Needs Work (<40%)')],
        loc='lower right', fontsize=7, facecolor='#1e293b', edgecolor='#334155', labelcolor='#94a3b8')
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    # ── Recommendations ──────────────────────────────────────
    st.markdown('<div class="section-header">Personalised Recommendations</div>', unsafe_allow_html=True)
    recs = []
    if orig_caf >= 100:   recs.append("☕ Reduce caffeine intake — high caffeine significantly disrupts sleep quality.")
    if orig_alc >= 3:     recs.append("🍷 Cut back on alcohol — it fragments sleep and reduces sleep quality.")
    if orig_smk == "Yes": recs.append("🚬 Smoking is linked to poor sleep. Consider speaking to a professional about quitting.")
    if orig_exe <= 1:     recs.append("🏃 Aim for at least 3 days of exercise per week — it improves sleep depth.")
    if orig_awk >= 3:     recs.append("😴 Frequent awakenings may indicate stress or sleep apnea — consider consulting a doctor.")
    if orig_sleep < 6:    recs.append("🕐 You're sleeping less than 6 hours. Aim for 7–9 hours for optimal health.")
    if orig_sleep > 10:   recs.append("🕐 Sleeping more than 10 hours can indicate underlying issues. Try maintaining a consistent schedule.")
    if not recs:
        if prediction == "Poor Sleep":
            recs += ["⚠️ Try maintaining a consistent sleep schedule and reducing stress.",
                     "📵 Avoid screens at least 1 hour before bed.",
                     "🧘 Try meditation or deep breathing before sleep."]
        elif prediction == "Moderate Sleep":
            recs += ["📋 Try setting a consistent bedtime every night.",
                     "🌿 Consider light morning exercise to improve your sleep cycle."]
        else:
            recs.append("🌟 Great job! Keep maintaining your healthy lifestyle habits.")
    for rec in recs:
        st.markdown(f'<div class="rec-card">{rec}</div>', unsafe_allow_html=True)

    # ── Compare to Healthy Sleeper ───────────────────────────
    st.markdown('<div class="section-header">📊 You vs. Average Good Sleeper</div>', unsafe_allow_html=True)
    st.markdown('<div class="rec-card" style="border-left-color:#10b981;margin-bottom:1rem;">See how your habits compare to the average person with <strong>Good Sleep</strong> in our dataset.</div>', unsafe_allow_html=True)

    healthy_avg = {
        "Sleep Duration (hrs)":  7.44,
        "Awakenings":            0.49,
        "Caffeine (mg/day)":     27.23,
        "Alcohol (drinks/wk)":   0.72,
        "Exercise (days/wk)":    2.37,
    }
    user_vals = {
        "Sleep Duration (hrs)":  orig_sleep,
        "Awakenings":            orig_awk,
        "Caffeine (mg/day)":     orig_caf,
        "Alcohol (drinks/wk)":   orig_alc,
        "Exercise (days/wk)":    orig_exe,
    }
    lower_is_better = {"Awakenings", "Caffeine (mg/day)", "Alcohol (drinks/wk)"}

    comp_labels = list(healthy_avg.keys())
    x = np.arange(len(comp_labels)); width = 0.35
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    fig3.patch.set_facecolor('#1e293b'); ax3.set_facecolor('#1e293b')
    ax3.bar(x - width/2, list(healthy_avg.values()), width, label='Avg Good Sleeper', color='#10b981', alpha=0.85)
    ax3.bar(x + width/2, [user_vals[l] for l in comp_labels], width, label='You', color='#3b82f6', alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels(comp_labels, rotation=0, ha='center', color='#94a3b8', fontsize=7)
    ax3.tick_params(axis='y', colors='#94a3b8', labelsize=8)
    ax3.spines['bottom'].set_color('#334155'); ax3.spines['left'].set_color('#334155')
    ax3.spines['top'].set_visible(False);     ax3.spines['right'].set_visible(False)
    ax3.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#94a3b8', fontsize=8)
    ax3.set_title("Your Habits vs. Average Good Sleeper", color='#e2e8f0', fontsize=10, pad=10)
    plt.tight_layout(); st.pyplot(fig3); plt.close()

    insights = []
    for label in comp_labels:
        u = user_vals[label]; h = healthy_avg[label]
        if label in lower_is_better:
            if u > h + 0.5:   insights.append(f"⚠️ Your <strong>{label}</strong> ({u}) is higher than good sleepers ({h}) — consider reducing it.")
            elif u < h - 0.5: insights.append(f"✅ Your <strong>{label}</strong> ({u}) is lower than good sleepers ({h}) — great!")
        else:
            if u < h - 0.5:   insights.append(f"⚠️ Your <strong>{label}</strong> ({u}) is lower than good sleepers ({h}) — try to increase it.")
            else:              insights.append(f"✅ Your <strong>{label}</strong> ({u}) is on par or better than good sleepers ({h})!")
    for insight in insights[:4]:
        st.markdown(f'<div class="rec-card">{insight}</div>', unsafe_allow_html=True)

    # ── PDF Download ─────────────────────────────────────────
    st.markdown('<div class="section-header">📄 Download Your Sleep Report</div>', unsafe_allow_html=True)
    orig_inputs = {
        "Age": orig_age, "Gender": orig_gender,
        "Sleep Duration (hrs)": orig_sleep,
        "Awakenings": orig_awk,
        "Caffeine (mg/day)": orig_caf,
        "Alcohol (drinks/wk)": orig_alc,
        "Smoking": orig_smk,
        "Exercise (days/wk)": orig_exe,
    }
    pdf_buffer = generate_pdf(prediction, score, orig_inputs, recs, insights)
    st.download_button(
        label="📥 Download PDF Report",
        data=pdf_buffer,
        file_name="SleepWise_Report.pdf",
        mime="application/pdf"
    )
