import streamlit as st
import cv2
import numpy as np
import pickle
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops


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

st.title("Prediksi Makanan Berkuah / Tidak Berkuah 🍲")
uploaded_file = st.file_uploader("Upload gambar makanan", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Gambar yang di-upload", use_container_width=True)

    
    # Ekstrak fitur
    extractor = feature_extractors[extractor_name]
    features = extractor(img).reshape(1,-1)
    features = scaler.transform(features)
    
    # Prediksi
    pred = model.predict(features)[0]
    label = "Berkuah 🍲" if pred==0 else "Tidak Berkuah 🥗"
    st.success(f"Hasil prediksi: {label}")
