# automatic-attendance-monitoring-system

A Streamlit-based offline/local face-attendance app.
Detects faces with SCRFD and computes face embeddings (default: InsightFace — buffalo_l; optional: FaceNet / facenet-pytorch).
Enroll students from photos (name, SAP ID, email) to build per-student prototype embeddings, then mark attendance from a group photo with manual verification and CSV/annotated-image export.

> **Note:** This repository contains code only. The app creates and stores runtime data under `streamlit_attendance_data/`. Do **not** commit that folder or any student images to version control.

---

## Features
- Enroll students from multiple photos to compute a per-student prototype embedding.
- Detect faces in a class/group photo and suggest matches based on cosine similarity.
- Manual review UI to correct assignments before finalizing attendance.
- Export attendance CSV and a visualization image that marks present/unknown faces and lists absentees.
- Streamlit UI for quick local testing and lightweight deployment.

---

## Development status

**Under active development.** This project is not yet feature-complete and will receive multiple future updates. Planned and possible future improvements include (but are not limited to):

- Improved face-detection robustness (alternative detectors / model upgrades).
- Liveness/detection-of-spoofing to reduce false accepts from photos/video.
- Better alignment / augmentation during enrollment to improve prototype quality.
- Batch processing optimizations and GPU support for faster embedding extraction.
- A REST API wrapper for programmatic attendance ingestion/use.
- Multi-camera / video stream support (real-time attendance).
- Role-based access control and authentication for deployments.
- Data encryption at rest and audit logging for privacy compliance.
- Dockerfile and CI (GitHub Actions) for easier deployment and automated testing.
- Unit tests, integration tests, and continuous integration pipelines.
- UI/UX refinements, accessibility improvements, and localization.
- Optional cloud storage integration (S3) and handling for large datasets.
- Documentation, example datasets (anonymized), and usage guides.

If you plan to use this in production or share with others, treat the current code as experimental and test carefully. Contributions and suggestions are welcome!

---

## Requirements

streamlit
numpy
opencv-python
Pillow
pandas
scikit-learn
scrfd
insightface
onnxruntime
facenet-pytorch    # if you want to use FaceNet alternative

---

## Installation (Local)

```bash
# clone repo
git clone https://github.com/void-desperado/automatic-attendance-monitoring-system.git
cd automatic-attendance-monitoring-system

# create venv and activate (Linux/macOS)
python -m venv .venv
source .venv/bin/activate

# Windows (Powershell)
# python -m venv .venv
# .venv\Scripts\Activate.ps1

pip install -U pip
pip install -r requirements.txt
```

---

## Run the app

```
streamlit run app.py
```
Open the local URL shown by Streamlit (usually http://localhost:8501).

## Acknowledgements

SCRFD models and implementations (face detection)
InsightFace / ArcFace and facenet-pytorch communities (face embeddings)
Streamlit for the easy web UI
