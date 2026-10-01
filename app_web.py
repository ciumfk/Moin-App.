import streamlit as st
import cv2
import mediapipe as mp
import pickle
import numpy as np
from PIL import Image
import time
import os

# ضبط إعدادات الصفحة
st.set_page_config(page_title="مُعين", page_icon="✋", layout="centered")

# تنسيق CSS لضبط الألوان البيضاء والأزرار الزرقاء بنسبة 3:4 والتصميم
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@700&display=swap');
    html, body, [class*="css"] { font-family: 'Tajawal', sans-serif; direction: rtl; text-align: right; }
    .stApp { background-color: white; }
    .main-title { text-align: center; font-size: 48px; color: #1e293b; font-weight: bold; margin-bottom: 40px; }
    .stButton>button {
        background-color: #87CEEB !important;
        color: black !important;
        font-size: 20px !important;
        font-weight: bold !important;
        border-radius: 15px !important;
        width: 100% !important;
        height: 150px !important; /* نسبة تقريبية 3 إلى 4 */
        border: none !important;
    }
    .back-btn>button {
        height: 50px !important;
        width: 100px !important;
        font-size: 14px !important;
    }
</style>
""", unsafe_allow_html=True)

# تحميل النموذج بمسار نسبي متوافق مع الويب وويندوز
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, 'sign_model.p')
    if os.path.exists(model_path):
        return pickle.load(open(model_path, 'rb'))
    return None

model = load_model()

# إدارة التنقل بين الصفحات
if 'page' not in st.session_state:
    st.session_state.page = 'main'

def set_page(page_name):
    st.session_state.page = page_name

# 1. الشاشة الرئيسية
if st.session_state.page == 'main':
    st.markdown("<div class='main-title'>مُعين</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("ترجمة لغة الإشارة"):
            set_page('cam')
            st.rerun()
    with col2:
        if st.button("تحدث معهم"):
            set_page('talk')
            st.rerun()

# 2. صفحة ترجمة لغة الإشارة (الكاميرا)
elif st.session_state.page == 'cam':
    st.markdown("<h2 style='text-align: center; color: #1e293b;'>مُعين - ترجمة لغة الإشارة</h2>", unsafe_allow_html=True)
    st.info("💡 اضغطي على زر بدء الكاميرا للبدء")
    
    # خيار تشغيل الكاميرا من المتصفح مباشرة
    img_file = st.camera_input("التقطي إشارة اليد أو شغلي الكاميرا")
    
    if img_file and model:
        # معالجة الصورة بنفس دقة النموذج
        bytes_data = img_file.getvalue()
        cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
        res = hands.process(rgb)
        
        if res.multi_hand_landmarks:
            for lm in res.multi_hand_landmarks:
                data_raw = []
                for p in lm.landmark: data_raw.extend([p.x, p.y, p.z])
                if len(data_raw) == 63:
                    probs = model.predict_proba([data_raw])[0]
                    detected_char = model.classes_[np.argmax(probs)]
                    st.success(f"الحرف المكتشف: **{detected_char}**")
        else:
            st.warning("لم يتم التعرّف على اليد، حاول تقريب اليد من الكاميرا.")

    st.markdown("---")
    st.markdown("<div class='back-btn'>", unsafe_allow_html=True)
    if st.button("الرجوع"):
        set_page('main')
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# 3. صفحة تحدث معهم
elif st.session_state.page == 'talk':
    st.markdown("<h2 style='text-align: center; color: #1e293b;'>مُعين - تحدث معهم</h2>", unsafe_allow_html=True)
    
    col_txt, col_chk = st.columns([4, 1])
    with col_txt:
        text_in = st.text_input("اكتب السطر هنا...", key="talk_in")
    with col_chk:
        st.write(" ")
        st.write(" ")
        start_btn = st.button("✔")
        
    hand_spot = st.empty()
    
    if start_btn and text_in:
        chars = [c for c in text_in if c != ' ']
        for char in chars:
            hand_spot.markdown(f"""
            <div style='text-align: center; background: #f8fafc; padding: 30px; border-radius: 15px;'>
                <span style='font-size: 80px;'>✋</span><br>
                <h3 style='color: #1e293b;'>حرف: {char}</h3>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1)
            
    st.markdown("---")
    if st.button("انتهى"):
        hand_spot.empty()
        
    st.markdown("<div class='back-btn'>", unsafe_allow_html=True)
    if st.button("الرجوع"):
        set_page('main')
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)