#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# إعداد الصفحة
st.set_page_config(page_title="تحليل أداء Valorant", layout="wide")

# ترويسة مميزة
st.markdown("""
    <h1 style='text-align: center; color: #FF4655;'>تحليل أداء لاعبي Valorant - Aura</h1>
    <p style='text-align: center; color: gray;'>تقرير تفاعلي يعتمد على بيانات حقيقية وتوقعات ذكية لتحسين أداء اللاعبين</p>
    <hr style='border: 1px solid #FF4655;'>
""", unsafe_allow_html=True)

# قراءة البيانات
try:
    data = pd.read_csv("player_data.csv", sep=";")
    data.columns = data.columns.str.strip()

    # تنظيف الأعمدة
    data["HS%"] = data["HS%"].str.replace("%", "").astype(float) / 100
    data["KAST"] = data["KAST"].str.replace("%", "").astype(float) / 100

    if "Player" not in data.columns:
        st.error("العمود 'Player' غير موجود.")
    else:
        st.success("تم تحميل البيانات بنجاح!")

        # عرض البيانات
        with st.expander("عرض بيانات اللاعبين"):
            st.dataframe(data)

        # تحليل الأداء
        st.markdown("### تحليل نقاط القوة والضعف:")
        for _, row in data.iterrows():
            with st.expander(f"اللاعب: {row['Player']}"):
                if row["K/D"] > 1.3:
                    st.success("نقطة قوة: أداء قتالي ممتاز!")
                else:
                    st.warning("نقطة ضعف: حاول تقلل عدد الوفيات.")

                if row["HS%"] > 0.28:
                    st.success("نقطة قوة: دقة رماية عالية!")
                else:
                    st.info("اقتراح: درّب على تحسين التصويب.")

                if row["FK"] > row["FD"]:
                    st.success("نقطة قوة: مبادرة قوية في الجولات!")
                else:
                    st.info("اقتراح: كن أكثر حذرًا في بداية الجولات.")

                if row["KAST"] > 0.73:
                    st.success("نقطة قوة: مساهمة ممتازة!")
                else:
                    st.info("اقتراح: عزز مشاركتك في الجولات.")

        # رسوم بيانية
        st.markdown("### المقارنات البصرية:")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("K/D Ratio")
            st.bar_chart(data.set_index("Player")["K/D"])

        with col2:
            st.subheader("Headshot %")
            st.bar_chart(data.set_index("Player")["HS%"])

        # جزء التعلم الآلي
        st.markdown("### توقع الأداء العام بالذكاء الاصطناعي:")

        # تصنيف الأداء
        data["Performance"] = data["K/D"].apply(lambda x: "Excellent" if x > 1.3 else "Needs Improvement")

        features = data[["K/D", "HS%", "KAST", "FK", "FD"]]
        target = data["Performance"]

        le = LabelEncoder()
        y = le.fit_transform(target)

        model = LogisticRegression()
        model.fit(features, y)

        preds = model.predict(features)
        data["Predicted Performance"] = le.inverse_transform(preds)

        st.dataframe(data[["Player", "Performance", "Predicted Performance"]])

except FileNotFoundError:
    st.error("ملف player_data.csv غير موجود.")
except Exception as e:
    st.error(f"صار خطأ: {str(e)}")





# إدخال لاعب جديد
st.markdown("## أضف أداء لاعب جديد")
with st.form("new_player_form"):
    new_kd = st.number_input("K/D", min_value=0.0, max_value=5.0, step=0.01)
    new_hs = st.slider("Headshot %", 0.0, 1.0, step=0.01)
    new_kast = st.slider("KAST", 0.0, 1.0, step=0.01)
    new_fk = st.number_input("First Kills", min_value=0)
    new_fd = st.number_input("First Deaths", min_value=0)
    submitted = st.form_submit_button("حلل الأداء")

if submitted:
    new_data = pd.DataFrame([{
        "K/D": new_kd,
        "HS%": new_hs,
        "KAST": new_kast,
        "FK": new_fk,
        "FD": new_fd
    }])

    prediction = model.predict(new_data)
    predicted_label = le.inverse_transform(prediction)[0]

    st.markdown(f"### التقييم العام: **{predicted_label}**")

    st.markdown("### نصائح مخصصة:")
    if new_kd > 1.3:
        st.success("ممتاز! أداءك القتالي عالي جدًا.")
    else:
        st.warning("حاول تقلل الوفيات وتحسن معدل القتل.")

    if new_hs > 0.28:
        st.success("تصويبك دقيق! استمر.")
    else:
        st.info("جرب تتدرب على الـ aim maps لزيادة الدقة.")

    if new_fk > new_fd:
        st.success("بداية قوية في الجولات!")
    else:
        st.info("انتبه من الاندفاع الزايد في البداية.")

    if new_kast > 0.73:
        st.success("مشاركتك في الجولات ممتازة.")
    else:
        st.info("حاول تساهم أكثر في كل جولة.")




# In[1]:


#get_ipython().system('jupyter nbconvert --to script valorant.ipynb')


# In[ ]:




