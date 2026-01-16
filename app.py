import streamlit as st
import cv2
import numpy as np
import pickle
import time
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops

st.set_page_config(page_title="Robobite Delivery System", page_icon="🤖")

# Load model
with open("model_makanan_full.pkl", "rb") as f:
    data = pickle.load(f)
model = data['model']
scaler = data['scaler']
extractor_name = data['feature_extractor']

def extract_lbp_features(image):
    resized = cv2.resize(image, (128,128))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 24, 3, method='uniform')  # pakai skimage
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 26), range=(0, 25))
    hist = hist.astype("float") / (hist.sum() + 1e-7)
    return hist

def extract_color_features(image):
    resized = cv2.resize(image, (128,128))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    feats = [
        np.mean(h), np.std(h), np.mean(h.flatten()), np.std(h.flatten()),
        np.mean(s), np.std(s), np.mean(s.flatten()), np.std(s.flatten()),
        np.mean(v), np.std(v), np.mean(v.flatten()), np.std(v.flatten())
    ]
    return np.array(feats)

def extract_haralick_features_simple(image):
    resized = cv2.resize(image, (128,128))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    feats = [graycoprops(glcm, p)[0,0] for p in props]
    return np.array(feats)

def extract_combined_features(image):
    return np.hstack([
        extract_lbp_features(image),
        extract_color_features(image),
        extract_haralick_features_simple(image)
    ])

# Mapping extractor
feature_extractors = {
    "Combined": extract_combined_features
}
st.title("Robobite: Smart Food Delivery 🤖🍲")
st.markdown("Sistem klasifikasi makanan untuk mengatur kecepatan robot pengantar.")
# st.title("Prediksi Makanan Berkuah / Tidak Berkuah 🍲")
uploaded_file = st.file_uploader("Upload gambar makanan", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Gambar yang di-upload", use_container_width=True)

    
    # Ekstrak fitur
    extractor = feature_extractors[extractor_name]
    features = extractor(img).reshape(1,-1)
    features = scaler.transform(features)
    
    # Prediksi
    pred = model.predict(features)[0]
    # label = "Berkuah 🍲" if pred==0 else "Tidak Berkuah 🥗"
    # st.success(f"Hasil prediksi: {label}")
    with col2:
        st.subheader("Hasil Analisis")
        if pred == 0:
            label = "Berkuah 🍲"
            speed_text = "Lambat (Hati-hati tumpah!)"
            speed_delay = 5  # Lebih lambat
            st.warning(f"Prediksi: **{label}**")
        else:
            label = "Tidak Berkuah 🥗"
            speed_text = "Normal/Sedikit Cepat"
            speed_delay = 3  # Lebih cepat
            st.success(f"Prediksi: **{label}**")
        
        st.info(f"Mode Jalan: **{speed_text}**")
    st.divider()
    
    # Placeholder untuk teks status dan progress bar
    status_text = st.empty()
    status_box = st.empty()
    progress_bar = st.progress(0)
    robot_icon = st.empty()

    for percent_complete in range(101):
        status_box.markdown(
            f"""
            <div style="background-color: #fcf3cf; padding: 15px; border-radius: 5px; border-left: 5px solid #f1c40f;">
                <span style="color: #9a7d0a; font-weight: bold;">🚚 Status: Sedang dalam perjalanan...</span>
            </div>
            """, 
            unsafe_allow_html=True
        )
        # Animasi sederhana menggunakan emoji
        distance = percent_complete // 5
        robot_line = " " * distance + "🤖" + "—" * (20 - distance) + " 🏁 (Meja)"
        robot_icon.text(robot_line)
        
        status_text.text(f"Progress Pengantaran: {percent_complete}%")
        progress_bar.progress(percent_complete)
        
        # Pengaturan kecepatan berdasarkan jenis makanan
        time.sleep(speed_delay)
    status_box.markdown(
        f"""
        <div style="background-color: #d4edda; padding: 15px; border-radius: 5px; border-left: 5px solid #28a745;">
            <span style="color: #155724; font-weight: bold;">✅ Status: Sampai di tujuan!</span>
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.balloons()
    st.success("✅ Pesanan telah sampai di meja tujuan dengan aman!")
