import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# =============== إعداد الواجهة والألوان ===============
st.set_page_config(
    page_title="نظام AURA - محلل أداء فالورانت PRO",
    page_icon="🎯",
    layout="wide"
)

# CSS مخصص للألوان والتنسيق
st.markdown("""
<style>
    /* تنسيق عام للنص */
    body {
        color: #FFFFFF !important;
    }
    
    /* عناوين رئيسية */
    h1, h2, h3 {
        color: #FF4655 !important;
        font-family: 'Arial', sans-serif;
    }
    
    /* صناديق المعلومات */
    .info-box {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #FF4655;
    }
    
    /* الأزرار */
    .stButton>button {
        background-color: #FF4655;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
    }
    
    /* الشرائح */
    .stSlider>div>div>div>div {
        background-color: #FF4655 !important;
    }
    
    /* الجداول */
    .stDataFrame {
        background-color: #1E1E1E;
    }
    
    /* علامات التبويب */
    .stTabs>div>div>button {
        color: #FF4655 !important;
    }
</style>
""", unsafe_allow_html=True)

# =============== تحميل البيانات ===============
@st.cache_data
def load_data():
    # بيانات افتراضية لـ 8 لاعبين (يمكن استبدالها ببيانات حقيقية)
    data = {
        "Player": ["Player1", "Player2", "Player3", "Player4", "Player5", "Player6", "Player7", "Player8"],
        "K/D": [1.45, 1.67, 0.98, 1.25, 1.53, 1.12, 1.89, 0.87],
        "HS%": [0.32, 0.41, 0.18, 0.25, 0.29, 0.22, 0.38, 0.15],
        "KAST": [0.78, 0.85, 0.65, 0.72, 0.75, 0.68, 0.82, 0.60],
        "FK": [20, 25, 8, 15, 18, 12, 22, 7],
        "FD": [10, 8, 15, 12, 11, 14, 9, 16],
        "Agent": ["Jett", "Phoenix", "Sage", "Reyna", "Omen", "Brimstone", "Raze", "Cypher"]
    }
    df = pd.DataFrame(data)
    df["Performance"] = df["K/D"].apply(lambda x: "Excellent" if x > 1.3 else "Needs Improvement")
    return df

data = load_data()

# =============== واجهة المستخدم ===============
st.title("🎮 نظام AURA PRO - محلل أداء فالورانت")
st.markdown("""
<div class="info-box">
    <p>نظام متكامل يستخدم الذكاء الاصطناعي لتحليل أداء اللاعبين وتقديم توصيات مخصصة</p>
</div>
""", unsafe_allow_html=True)

# =============== قسم التحليلات ===============
st.header("📊 لوحة التحليلات")

# تحليل إحصائي
st.subheader("الإحصائيات الرئيسية")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("متوسط K/D", f"{data['K/D'].mean():.2f}", delta="+0.15 عن المعدل العالمي")
with col2:
    st.metric("أعلى HS%", f"{data['HS%'].max():.1%}", delta=data.loc[data['HS%'].idxmax(), 'Player'])
with col3:
    st.metric("أفضل لاعب", data.loc[data['K/D'].idxmax(), 'Player'], delta=f"K/D: {data['K/D'].max():.2f}")
with col4:
    st.metric("عدد اللاعبين", len(data), delta="+2 عن الإصدار السابق")

# تحليل الأداء
st.subheader("توزيع الأداء")
tab1, tab2 = st.tabs(["مخطط شريطي", "مخطط دائري"])
with tab1:
    fig = px.bar(data, x="Player", y="K/D", color="Performance", 
                 color_discrete_map={"Excellent": "#4CAF50", "Needs Improvement": "#FF9800"})
    st.plotly_chart(fig, use_container_width=True)
with tab2:
    fig = px.pie(data, names="Performance", color="Performance",
                 color_discrete_map={"Excellent": "#4CAF50", "Needs Improvement": "#FF9800"})
    st.plotly_chart(fig, use_container_width=True)

# =============== قسم الذكاء الاصطناعي ===============
st.header("🤖 محرك الذكاء الاصطناعي")

# تحضير البيانات
features = data[["K/D", "HS%", "KAST", "FK", "FD"]]
target = LabelEncoder().fit_transform(data["Performance"])

# اختيار النموذج
model_type = st.radio("اختر خوارزمية الذكاء الاصطناعي", 
                     ["Random Forest", "Gradient Boosting"], horizontal=True)

if model_type == "Random Forest":
    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    )
else:
    model = make_pipeline(
        StandardScaler(),
        GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42)
    )

# تدريب النموذج
cv_scores = cross_val_score(model, features, target, cv=KFold(5, shuffle=True))
model.fit(features, target)

st.markdown(f"""
<div class="info-box">
    <h4>أداء النموذج</h4>
    <p><b>الدقة:</b> {cv_scores.mean():.1%} (± {cv_scores.std():.1%})</p>
    <p><b>الخوارزمية:</b> {model_type}</p>
    <p><b>عدد العينات:</b> {len(data)} لاعب</p>
</div>
""", unsafe_allow_html=True)

# =============== محلل الأداء الفوري ===============
st.header("🔮 محلل الأداء الفوري")

with st.form("player_form"):
    st.subheader("أدخل إحصائيات اللاعب")
    
    c1, c2 = st.columns(2)
    with c1:
        kd = st.slider("نسبة القتل/الموت (K/D)", 0.5, 3.0, 1.3, 0.05)
        hs = st.slider("نسبة تصويب الرأس (HS%)", 0.1, 0.5, 0.25, 0.01)
    with c2:
        kast = st.slider("معدل المساهمة (KAST)", 0.5, 1.0, 0.7, 0.01)
        fk = st.number_input("القتل الأول (FK)", 0, 30, 10)
        fd = st.number_input("الموت الأول (FD)", 0, 30, 10)
    
    submitted = st.form_submit_button("حلل الأداء")

if submitted:
    new_data = [[kd, hs, kast, fk, fd]]
    pred = model.predict(new_data)
    proba = model.predict_proba(new_data)[0]
    
    if pred[0] == 1:
        st.success("### التقييم: أداء ممتاز! 🏆")
        st.metric("ثقة النموذج", f"{proba[1]:.1%}")
    else:
        st.error("### التقييم: يحتاج تحسين 🛠")
        st.metric("ثقة النموذج", f"{proba[0]:.1%}")
    
    # النصائح المخصصة
    st.subheader("💡 توصيات تطوير الأداء")
    if kd < 1.0:
        st.warning("- ركز على تحسين قراراتك القتالية وتجنب المواقف الخطرة")
    if hs < 0.2:
        st.warning("- تدرب يومياً على تمارين التصويب للرأس لمدة 15 دقيقة")
    if kast < 0.7:
        st.warning("- حاول البقاء حياً لفترة أطول في الجولات")
    if fd > fk:
        st.warning("- تحلى بالصبر في بداية الجولات وتجنب المخاطرة غير المحسوبة")

# =============== قسم المعلومات الجانبية ===============
with st.sidebar:
    st.image("https://i.imgur.com/3JQ2X1a.png", width=200)  # يمكن استبدالها بشعارك
    st.title("إعدادات النظام")
    
    st.markdown("""
    ### معايير التقييم:
    - *ممتاز*: 
      - K/D > 1.3
      - HS% > 28%
      - KAST > 73%
    - *جيد*: بين المعايير
    - *ضعيف*: أقل من المعايير
    """)
    
    st.markdown("""
    ### خصائص الإصدار:
    - إصدار: AURA PRO 2.1
    - تاريخ التحديث: 2023-10-15
    - المطور: فريق AURA
    """)
    
    if st.button("🔄 تحديث البيانات"):
        st.cache_data.clear()
        st.experimental_rerun()

# =============== نهاية التطبيق ===============
st.markdown("---")
st.markdown("""
<p style="text-align: center; color: #777;">
    نظام AURA  © 2025 | جميع الحقوق محفوظة
</p>
""", unsafe_allow_html=True)
