import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os
import base64 

def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
    
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{get_base64_of_image('background.jpg')}");
        background-size: cover;
        background-position: center;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(
    page_title="Детектор болезней риса",
    page_icon="🌾",
    layout="wide"
)

@st.cache_resource
def load_model():
    model_path = 'best.pt'
    if not os.path.exists(model_path):
        st.error("❌ Модель не найдена! Поместите файл best.pt в указанную директорию.")
        return None
    return YOLO(model_path)

model = load_model()

st.title("🌾 Детектор болезней риса")
st.write("Загрузите изображение листа риса, чтобы определить наличие заболевания.")

uploaded_file = st.file_uploader(
    "📤 Загрузите изображение", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Исходное изображение", use_container_width=True)

    if model:
        img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        with st.spinner("🔎 Анализ изображения..."):
            results = model.predict(
                source=img_array,
                imgsz=640,
                conf=0.25,
                save=False
            )

        boxes = results[0].boxes
        names = model.names
        detected_diseases = []

        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)

        try:
            font = ImageFont.truetype("arial.ttf", size=18)
        except:
            font = ImageFont.load_default()

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{names[cls_id]} {conf:.2f}"
            detected_diseases.append(names[cls_id])

            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
            draw.text((x1, y1 - 15), label, fill="red", font=font)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Исходное изображение", use_container_width=True)
        with col2:
            st.image(annotated_image, caption="Результат детекции", use_container_width=True)

        if detected_diseases:
            st.warning("🚨 Обнаруженные болезни:")
            for disease in set(detected_diseases):
                st.write(f"- {disease}")
        else:
            st.success("✅ Болезни не обнаружены!")

        st.download_button(
            label="⬇️ Скачать результат",
            data=cv2.imencode('.jpg', cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR))[1].tobytes(),
            file_name="result.jpg",
            mime="image/jpeg"
        )

st.sidebar.title("О приложении")
st.sidebar.info("""
Это приложение использует модель YOLOv8 для детекции:
- Бактериального ожога листьев  
- Бурой пятнистости  
- Грибкового поражения листьев  
""")

st.sidebar.header("Примеры для теста")
sample_images = {
    "bacterial_leaf_blight": r"\data\images\val\blight_rotated_006_PNG_jpg.rf.be0c930f8a9de9185987076195260bf8.jpg",
    "brown_spot": r"data\images\val\Blast_1208_jpg.rf.fb486ad2e58f5be5edd77f76512b4906.jpg",
    "leaf_mold": r"data\images\val\blast_rotated_068_JPG_jpg.rf.4349e0cfef5c9e8a999d979cb9848882.jpg"
}

for name, path in sample_images.items():
    if os.path.exists(path):
        st.sidebar.image(path, caption=name, use_container_width=True)