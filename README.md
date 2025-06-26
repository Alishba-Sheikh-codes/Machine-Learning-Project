# 🧠 Brain Tumor Segmentation with YOLOv11 and SAM2 (Colab + Streamlit)

This project performs brain tumor detection and segmentation using **YOLOv11** (for object detection) and **SAM2** (Segment Anything Model 2) for precise segmentation. It includes a deployable **Streamlit app** that runs directly in Google Colab and is shared via a secure **Cloudflare tunnel**.

---

## 🔍 Objective

- Detect brain tumors in MRI images using YOLOv11.
- Segment the tumor boundaries using SAM2.
- Provide a user-friendly interface using Streamlit.

---

## 📁 Folder Structure

/content/SAM_YOLO11/
├── datasets/
│ ├── images/
│ ├── labels/
│ └── data.yaml
├── runs/
├── app.py # Streamlit app file
├── yolo11s.pt # YOLOv11 model checkpoint
├── sam2_b.pt # SAM2 model checkpoint


## 🧪 Environment Setup (Colab)

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy project files from Drive
!cp -r "/content/drive/MyDrive/SAM_YOLO11" /content/

# Install dependencies
!pip install ultralytics
!pip install -q streamlit opencv-python Pillow
## 🧠 Train YOLOv11 on Brain Tumor Dataset
!yolo detect train \
  data=/content/SAM_YOLO11/datasets/data.yaml \
  model=yolo11s.pt \
  epochs=15 \
  imgsz=480
## 🔍 Predict with YOLOv11
!yolo detect predict \
  model=/content/runs/detect/train/weights/best.pt \
  source=/content/SAM_YOLO11/datasets/test/images \
  save=True
⚙️ Streamlit App for Detection + Segmentation
The app (app.py) lets users upload an MRI image. YOLOv11 detects the tumor, then SAM2 segments it.

yolo_model = YOLO("yolo11s.pt")
sam_model = SAM("sam2_b.pt")

# Upload MRI image
uploaded_file = st.file_uploader("Upload MRI Image of brain")

# Run detection and segmentation
results = yolo_model(tmp_path)
boxes = results[0].boxes.xyxy.cpu()
sam_results = sam_model(results[0].orig_img, bboxes=boxes)
## 🌐 Launch Streamlit App in Colab
bash
Copy
Edit
# Download and run cloudflared
!wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared

# Run the app
!streamlit run app.py > streamlit_log.txt 2>&1 &
!sleep 5 && tail -n 20 streamlit_log.txt

# Start cloudflared tunnel
!./cloudflared tunnel --url http://localhost:8501
⏳ Wait for the link in Colab output to access your app.

## 🖼️ Features
Upload any brain MRI image.

Get tumor detection boxes from YOLOv11.

Get precise segmentation masks from SAM2.

Easy to use — runs in a browser.

## 🧠 Models Used
YOLOv11

Segment Anything Model (SAM2)

## 📌 Notes
Make sure to place your trained YOLO model (best.pt) and sam2_b.pt in the same working directory.

App uses GPU easy available in Colab (can be adapted for CPU but it is faster in GPU).
# Machine-Learning-Project
Brain Tumor segmentation using YOLLO11 and SAM2
