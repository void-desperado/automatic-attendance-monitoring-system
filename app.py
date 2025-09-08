# app.py
import os
import io
import json
import time
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2
import torch
from PIL import Image
import streamlit as st
from retinaface import RetinaFace
from facenet_pytorch import InceptionResnetV1

# -------------------------
# Config / filepaths
# -------------------------
DATA_DIR = Path("face_attendance_data")
DATA_DIR.mkdir(exist_ok=True)
PROTOS_PATH = DATA_DIR / "prototypes.npz"
DB_JSON = DATA_DIR / "students_db.json"
ATTENDANCE_CSV = DATA_DIR / "attendance_full.csv"
VISUALIZATION_IMG = DATA_DIR / "attendance_visualized.jpg"

IMG_SIZE = 160
BATCH_SIZE = 16
THRESHOLD_DEFAULT = 0.45

# alignment target points (must match enrollment)
DESIRED_LEFT_EYE = (44, 56)
DESIRED_RIGHT_EYE = (116, 56)
DESIRED_NOSE = (80, 100)

device = "cpu"
torch.set_num_threads(4)

# -------------------------
# Utility functions
# -------------------------
@st.cache_resource
def load_model():
    # loads facenet model (cached by streamlit)
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return model

MODEL = load_model()

def load_prototypes():
    if PROTOS_PATH.exists():
        data = np.load(str(PROTOS_PATH), allow_pickle=True)
        ids = data["ids"].tolist()
        names = data["names"].tolist()
        protos = data["prototypes"].astype(np.float32)
        # safety normalize
        norms = np.linalg.norm(protos, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        protos = (protos / norms).astype(np.float32)
        return ids, names, protos
    else:
        return [], [], np.zeros((0, 512), dtype=np.float32)  # empty

def save_prototypes(ids: List[str], names: List[str], prototypes: np.ndarray):
    np.savez_compressed(str(PROTOS_PATH),
                        ids=np.array(ids, dtype=object),
                        names=np.array(names, dtype=object),
                        prototypes=prototypes.astype(np.float32))

def load_db():
    if DB_JSON.exists():
        with open(DB_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {"meta": {}, "students": {}}

def save_db(db):
    with open(DB_JSON, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)

def imgfile_to_bgr(img_file) -> np.ndarray:
    # img_file: a file-like object (BytesIO from streamlit uploader)
    img = Image.open(img_file).convert("RGB")
    arr = np.array(img)  # RGB
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr

def write_temp_image(bgr_img) -> str:
    tf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tf.name, bgr_img)
    tf.close()
    return tf.name

def pick_largest_detection(dets):
    if not dets:
        return None
    best = None; best_area = 0
    for d in dets.values():
        x1,y1,x2,y2 = d["facial_area"]
        area = max(0, x2-x1) * max(0, y2-y1)
        if area > best_area:
            best_area = area; best = d
    return best

def align_face(img_bgr: np.ndarray, det: dict) -> np.ndarray:
    landmarks = det.get("landmarks", {})
    x1,y1,x2,y2 = det["facial_area"]
    h,w = img_bgr.shape[:2]
    x1,y1,x2,y2 = map(int, [max(0,x1), max(0,y1), min(w-1,x2), min(h-1,y2)])
    if landmarks and "left_eye" in landmarks and "right_eye" in landmarks:
        left_eye = landmarks["left_eye"]; right_eye = landmarks["right_eye"]
        if "nose" in landmarks:
            third = landmarks["nose"]
        else:
            ml = landmarks.get("mouth_left",[0,0]); mr = landmarks.get("mouth_right",[0,0])
            third = ((ml[0]+mr[0])/2.0, (ml[1]+mr[1])/2.0)
        src = np.array([left_eye, right_eye, third], dtype=np.float32)
        dst = np.array([DESIRED_LEFT_EYE, DESIRED_RIGHT_EYE, DESIRED_NOSE], dtype=np.float32)
        try:
            M = cv2.getAffineTransform(src, dst)
            aligned = cv2.warpAffine(img_bgr, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            return aligned
        except Exception:
            pass
    pad_x = int(0.1 * (x2-x1)); pad_y = int(0.1 * (y2-y1))
    xx1 = max(0, x1 - pad_x); yy1 = max(0, y1 - pad_y); xx2 = min(w-1, x2 + pad_x); yy2 = min(h-1, y2 + pad_y)
    crop = img_bgr[yy1:yy2, xx1:xx2].copy()
    if crop.size == 0:
        return cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

def prewhiten(img_rgb):
    x = img_rgb.astype(np.float32) / 255.0
    return (x - 0.5) / 0.5

def emb_from_bgr(face_bgr: np.ndarray, model=MODEL) -> np.ndarray:
    img_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    img_rgb = prewhiten(img_rgb)
    t = torch.from_numpy(np.transpose(img_rgb, (2,0,1))).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out = model(t)
    emb = out.cpu().numpy()[0].astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb

def batch_embeddings(bgr_list: List[np.ndarray], model=MODEL, batch_size=BATCH_SIZE) -> np.ndarray:
    embs = []
    for i in range(0, len(bgr_list), batch_size):
        batch = bgr_list[i:i+batch_size]
        arr = np.stack([prewhiten(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in batch], axis=0)
        t = torch.from_numpy(np.transpose(arr, (0,3,1,2))).float().to(device)
        with torch.no_grad():
            out = model(t)
        out_np = out.cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(out_np, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        out_np = out_np / norms
        embs.append(out_np)
    return np.vstack(embs) if embs else np.zeros((0,512), dtype=np.float32)

# -------------------------
# Enrollment logic
# -------------------------
def enroll_student_from_upload(student_id: str, student_name: str, uploaded_files: List[io.BytesIO]):
    """Save uploaded images to data dir (optional), compute prototype and add to prototypes/db"""
    if not uploaded_files:
        raise ValueError("No images uploaded for enrollment.")
    db = load_db()
    ids, names, protos = load_prototypes()

    embeddings = []
    saved_paths = []
    for up in uploaded_files:
        # read and save temp
        bgr = imgfile_to_bgr(up)
        temp_path = write_temp_image(bgr)
        saved_paths.append(temp_path)
        # detect face
        try:
            dets = RetinaFace.detect_faces(temp_path)
        except Exception as e:
            st.warning(f"RetinaFace failed on an uploaded image: {e}")
            continue
        det = pick_largest_detection(dets) if dets else None
        if det is None:
            st.warning("No face detected in one uploaded image; skipping that file.")
            continue
        aligned = align_face(bgr, det)
        emb = emb_from_bgr(aligned)
        embeddings.append(emb)

    if not embeddings:
        st.error("No usable face embeddings extracted — enrollment aborted.")
        return False

    embs_arr = np.vstack(embeddings).astype(np.float32)
    proto = np.mean(embs_arr, axis=0)
    proto_norm = np.linalg.norm(proto)
    if proto_norm > 0:
        proto = proto / proto_norm
    else:
        proto = embs_arr[0] / (np.linalg.norm(embs_arr[0]) + 1e-12)

    # append to prototypes list
    ids.append(student_id)
    names.append(student_name)
    if protos.size == 0:
        protos_new = proto[None, :].astype(np.float32)
    else:
        protos_new = np.vstack([protos, proto.astype(np.float32)])
    save_prototypes(ids, names, protos_new)

    # update DB
    db = load_db()
    db.setdefault("students", {})
    db["students"][student_id] = {
        "id": student_id,
        "name": student_name,
        "enrolled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_images": len(embeddings)
    }
    save_db(db)
    st.success(f"Enrolled {student_name} (id={student_id}) with {len(embeddings)} embeddings.")
    # cleanup temp images
    for p in saved_paths:
        try:
            os.remove(p)
        except Exception:
            pass
    return True

# -------------------------
# Attendance logic
# -------------------------
def detect_faces_in_image_bgr(img_bgr: np.ndarray):
    # Save to temp and pass path to RetinaFace to be robust
    tmp_path = write_temp_image(img_bgr)
    try:
        dets = RetinaFace.detect_faces(tmp_path)
    except Exception as e:
        os.remove(tmp_path)
        raise e
    os.remove(tmp_path)
    return dets

def process_class_photo_and_suggest(photo_bgr: np.ndarray, threshold: float):
    # returns dict with detection crops, suggested matches (id,name,score)
    ids, names, protos = load_prototypes()
    dets = detect_faces_in_image_bgr(photo_bgr)
    if not dets:
        return [], [], []

    h,w = photo_bgr.shape[:2]
    det_items = list(dets.items())
    crops = []
    bboxes = []
    for _, det in det_items:
        bbox = det.get("facial_area", None)
        if bbox is None:
            continue
        x1,y1,x2,y2 = map(int, bbox)
        # clamp
        x1,y1,x2,y2 = max(0,x1), max(0,y1), min(w-1,x2), min(h-1,y2)
        aligned = align_face(photo_bgr, det)
        crops.append(aligned)
        bboxes.append([x1,y1,x2,y2])

    if not crops:
        return [], [], []

    embs = batch_embeddings(crops)
    # compute sims to prototypes: (M,D) x (D,N) -> (M,N)
    sims = embs.dot(protos.T) if protos.size else np.zeros((len(crops),0), dtype=np.float32)
    suggested = []
    for i in range(len(crops)):
        if protos.size == 0:
            suggested.append({"student_id": None, "name": None, "score": 0.0})
            continue
        best_idx = int(np.argmax(sims[i]))
        best_score = float(sims[i, best_idx])
        if best_score >= threshold:
            suggested.append({"student_id": ids[best_idx], "name": names[best_idx], "score": best_score})
        else:
            suggested.append({"student_id": None, "name": None, "score": best_score})
    return crops, bboxes, suggested

def finalize_attendance(matches: List[dict], proto_ids: List[str], proto_names: List[str], photo_bgr: np.ndarray, out_csv: Path, out_img: Path):
    """
    matches: list of dicts for each detection:
       {assigned_id: str or '', assigned_name: str or '', score: float, bbox: [x1,y1,x2,y2]}
    Writes CSV with rows for all enrolled students (present/absent) + unknowns; writes visualization image.
    """
    # compute present set
    present_ids = set([m["assigned_id"] for m in matches if m.get("assigned_id")])
    # build rows
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    # present rows
    for sid in present_ids:
        # find first match for sid to fetch score and bbox
        m = next((x for x in matches if x.get("assigned_id")==sid), None)
        name = m.get("assigned_name") if m else ""
        score = m.get("score", 0.0) if m else 0.0
        bbox = m.get("bbox","") if m else ""
        rows.append({"timestamp": timestamp, "student_id": sid, "name": name, "score": score, "bbox": bbox, "status":"present"})
    # absent rows for everyone else
    for sid, name in zip(proto_ids, proto_names):
        if sid not in present_ids:
            rows.append({"timestamp": timestamp, "student_id": sid, "name": name, "score": 0.0, "bbox": "", "status":"absent"})
    # unknown rows: those matches with no assigned_id (but they are detections)
    for m in matches:
        if not m.get("assigned_id"):
            rows.append({"timestamp": timestamp, "student_id": "", "name": "", "score": m.get("score",0.0), "bbox": m.get("bbox",""), "status":"unknown"})
    # save CSV
    import csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp","student_id","name","score","bbox","status"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # visualization: draw boxes + absent panel
    vis = photo_bgr.copy()
    for m in matches:
        bbox = m.get("bbox", None)
        if not bbox:
            continue
        x1,y1,x2,y2 = bbox
        if m.get("assigned_id"):
            color = (0,255,0)
            label = f"{m.get('assigned_name')} {m.get('score',0.0):.2f}"
        else:
            color = (0,0,255)
            label = f"unknown {m.get('score',0.0):.2f}"
        cv2.rectangle(vis, (x1,y1),(x2,y2), color, 2)
        cv2.putText(vis, label, (x1, max(10,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    # absent panel
    absent_names = [n for i,n in zip(proto_ids, proto_names) if i not in present_ids]
    h_img = vis.shape[0]
    panel_w = 360
    panel = 255 * np.ones((h_img, panel_w, 3), dtype=np.uint8)
    pad = 8
    cv2.putText(panel, "ABSENT", (pad, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    y = 60
    line_h = 20
    max_lines = max(1, (h_img - 60) // line_h)
    for idx, nm in enumerate(absent_names[:max_lines]):
        cv2.putText(panel, f"{idx+1}. {nm}", (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        y += line_h
    if len(absent_names) > max_lines:
        cv2.putText(panel, f"... +{len(absent_names)-max_lines} more", (pad, y+6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    combined = np.concatenate([vis, panel], axis=1)
    cv2.imwrite(str(out_img), combined)
    return rows, combined

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Face Attendance", layout="wide")
st.title("Face Attendance — Enroll, Match, Review, Finalize")

menu = st.sidebar.selectbox("Choose action", ["Enroll student", "Take attendance", "Review last attendance"])

# load prototypes & db for UI display
proto_ids, proto_names, prototypes = load_prototypes()
db = load_db()

if menu == "Enroll student":
    st.header("Enroll a single student")
    with st.form("enroll_form"):
        student_id = st.text_input("Student ID (unique)", key="sid")
        student_name = st.text_input("Student name", key="sname")
        uploaded = st.file_uploader("Upload photos (multiple). Use clear frontal photos; upload 3-8 images recommended.", accept_multiple_files=True, type=["jpg","jpeg","png","bmp"])
        submit = st.form_submit_button("Enroll")
    if submit:
        if not student_id.strip():
            st.error("Student ID required")
        elif not uploaded:
            st.error("Upload at least one image")
        else:
            st.info("Enrolling... this may take a few seconds while embeddings are computed.")
            success = enroll_student_from_upload(student_id.strip(), student_name.strip() or student_id.strip(), uploaded)
            if success:
                # reload prototypes & db
                proto_ids, proto_names, prototypes = load_prototypes()
                db = load_db()
                st.success("Enrollment complete. Current enrolled students: " + str(len(proto_ids)))

    st.markdown("---")
    st.subheader("Enrolled students (sample)")
    if proto_ids:
        rows = [{"id":i,"name":n} for i,n in zip(proto_ids, proto_names)]
        st.dataframe(rows[:50])
    else:
        st.write("No students enrolled yet.")

elif menu == "Take attendance":
    st.header("Take attendance from a class photo")
    threshold = st.sidebar.slider("Matching threshold (cosine)", 0.0, 1.0, float(THRESHOLD_DEFAULT), 0.01)
    uploaded_photo = st.file_uploader("Upload class/group photo", type=["jpg","jpeg","png","bmp"])
    if uploaded_photo:
        photo_bgr = imgfile_to_bgr(uploaded_photo)
        st.image(cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB), caption="Uploaded photo", use_column_width=True)
        st.info("Detecting faces and suggesting matches...")

        crops, bboxes, suggested = process_class_photo_and_suggest(photo_bgr, threshold)
        if not crops:
            st.warning("No faces detected in the photo. Try a clearer photo or closer distance.")
        else:
            st.success(f"Detected {len(crops)} faces.")
            # Build UI to review each detection
            st.subheader("Review & correct matches")
            matches = []
            cols = st.columns(3)
            for i, (crop, bbox, sug) in enumerate(zip(crops, bboxes, suggested)):
                # show crop and suggestion
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                col = cols[i % 3]
                col.image(crop_rgb, width=200, caption=f"Face #{i+1}")
                if sug["student_id"]:
                    pre = f"{sug['student_id']} — {sug['name']} ({sug['score']:.3f})"
                else:
                    pre = f"Unknown (best score={sug['score']:.3f})"
                # options: predicted, any enrolled student, Unknown
                options = ["<Keep suggested>"] + [f"{sid} — {name}" for sid,name in zip(proto_ids, proto_names)] + ["<Unknown>"]
                sel = col.selectbox(f"Assign Face #{i+1}", options=options, index=0, key=f"match_sel_{i}")
                # process selection
                if sel == "<Keep suggested>":
                    assigned_id = sug["student_id"]
                    assigned_name = sug["name"]
                elif sel == "<Unknown>":
                    assigned_id = ""
                    assigned_name = ""
                else:
                    # parse selected string to id and name
                    assigned_id = sel.split(" — ", 1)[0]
                    assigned_name = sel.split(" — ", 1)[1] if " — " in sel else assigned_id
                matches.append({"assigned_id": assigned_id, "assigned_name": assigned_name, "score": sug["score"], "bbox": bbox})
            st.markdown("---")
            st.info("When you are happy with assignments, click Finalize attendance.")

            if st.button("Finalize attendance and save"):
                # finalize and save
                proto_ids_local, proto_names_local, protos_local = load_prototypes()
                rows_out, vis = finalize_attendance(matches, proto_ids_local, proto_names_local, photo_bgr, ATTENDANCE_CSV, VISUALIZATION_IMG)
                st.success(f"Attendance finalized. CSV saved to: {ATTENDANCE_CSV}; visualization saved to: {VISUALIZATION_IMG}")
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Attendance visualization (with absent panel)", use_column_width=True)
                st.markdown("## CSV preview")
                import pandas as pd
                df = pd.read_csv(ATTENDANCE_CSV)
                st.dataframe(df.head(200))

elif menu == "Review last attendance":
    st.header("Review last saved attendance (CSV + visualization)")
    if ATTENDANCE_CSV.exists():
        import pandas as pd
        df = pd.read_csv(ATTENDANCE_CSV)
        st.dataframe(df)
        st.write("Visualization image:")
        if VISUALIZATION_IMG.exists():
            vis = Image.open(VISUALIZATION_IMG)
            st.image(vis, use_column_width=True)
        else:
            st.write("No visualization image found.")
    else:
        st.write("No attendance CSV found yet. Run Take attendance first.")

# End of app
