# app.py
"""
Student Attendance Webapp (Streamlit)
Fix: correctly handle student folders named only by SAPID (numeric folder names).
Includes:
- SCRFD (ONNX) detector + InsightFace embeddings
- Enrollment (single student with name/sapid/email and ZIP bulk)
- Robust DB loader + migration (handles older .npz without sapids/emails)
- Attendance marking, annotated image, face preview grid, absent overlay
- Manual verification: inspect enrolled photos and mark absent as present
"""

import streamlit as st
from pathlib import Path
import tempfile, shutil, zipfile, json, re, os
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Folders & config ----------
BASE_DIR = Path.cwd()
APP_DATA = BASE_DIR / "streamlit_attendance_data"
MODELS_DIR = APP_DATA / "models"
ENROLL_DIR = APP_DATA / "enroll"
OUTPUT_DIR = APP_DATA / "output"
DB_PATH = APP_DATA / "enroll_db.npz"
SCRFD_DEFAULT_PATH = MODELS_DIR / "scrfd_500m.onnx"

for d in (APP_DATA, MODELS_DIR, ENROLL_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Attendance — SCRFD + ArcFace", layout="wide")
st.title("📸 Student Attendance — SCRFD + ArcFace")

st.sidebar.header("Models & Settings")
uploaded_scrfd = st.sidebar.file_uploader("Upload SCRFD ONNX (optional)", type=["onnx"])
if uploaded_scrfd is not None:
    model_path = MODELS_DIR / uploaded_scrfd.name
    with open(model_path, "wb") as f:
        f.write(uploaded_scrfd.read())
    st.sidebar.success(f"Saved SCRFD model to {model_path}")
    SCRFD_PATH = model_path
else:
    SCRFD_PATH = SCRFD_DEFAULT_PATH
    if SCRFD_PATH.exists():
        st.sidebar.info(f"Using SCRFD model: {SCRFD_PATH.name}")
    else:
        st.sidebar.warning(f"SCRFD ONNX not found at {SCRFD_PATH}. Upload in sidebar or place file there.")

SIM_THRESHOLD = st.sidebar.slider("Cosine similarity threshold (matching)", 0.20, 0.80, 0.45, 0.01)
DETECT_PROB = st.sidebar.slider("SCRFD detection probability threshold", 0.2, 0.8, 0.35, 0.01)
PAD_RATIO = st.sidebar.slider("Crop padding ratio (for enroll)", 0.0, 0.4, 0.12, 0.01)


# ---------- Cached model loading ----------
@st.cache_resource(show_spinner=False)
def init_models_cached(scrfd_path: str):
    from scrfd import SCRFD, Threshold
    detector = SCRFD.from_path(scrfd_path)
    threshold = Threshold(probability=DETECT_PROB)
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l")
    try:
        app.prepare(ctx_id=-1, det_size=(224,224))
    except TypeError:
        app.prepare(ctx_id=-1)
    return detector, threshold, app


# ---------- Robust DB loader & migration ----------
def robust_load_enroll_db(db_path: Path):
    if not db_path.exists():
        return {"names": [], "sapids": [], "emails": [], "embeddings": None}
    data = np.load(db_path, allow_pickle=True)
    files = set(data.files)
    names = list(data["names"].tolist()) if "names" in files else []
    embeddings = data["embeddings"] if "embeddings" in files else None
    sapids = list(data["sapids"].tolist()) if "sapids" in files else ["" for _ in names]
    emails = list(data["emails"].tolist()) if "emails" in files else ["" for _ in names]
    return {"names": names, "sapids": sapids, "emails": emails, "embeddings": embeddings}


def migrate_enroll_db_add_missing_keys(db_path: Path):
    if not db_path.exists():
        raise FileNotFoundError(f"No DB at {db_path}")
    data = np.load(db_path, allow_pickle=True)
    files = set(data.files)
    names = list(data["names"].tolist()) if "names" in files else []
    embeddings = data["embeddings"] if "embeddings" in files else None
    sapids = list(data["sapids"].tolist()) if "sapids" in files else ["" for _ in names]
    emails = list(data["emails"].tolist()) if "emails" in files else ["" for _ in names]
    if embeddings is None:
        embeddings = np.zeros((0,512), dtype=np.float32)
    tmp_path = db_path.with_suffix(".npz.tmp")
    np.savez_compressed(tmp_path,
                        names=np.array(names, dtype=object),
                        sapids=np.array(sapids, dtype=object),
                        emails=np.array(emails, dtype=object),
                        embeddings=embeddings)
    tmp_path.replace(db_path)
    return True


# ---------- Helpers ----------
def _extract_xyxy_from_bbox(bbox):
    try:
        import numpy as _np
        if isinstance(bbox, (list, tuple, _np.ndarray)):
            arr = _np.array(bbox)
            if arr.ndim == 1 and arr.size >= 4:
                a = arr.astype(float).ravel()[:4]
                if (a[2] > a[0]) and (a[3] > a[1]):
                    return int(a[0]), int(a[1]), int(a[2]), int(a[3])
                else:
                    return int(a[0]), int(a[1]), int(a[0]+a[2]), int(a[1]+a[3])
            if arr.ndim == 2 and arr.shape[0] == 2:
                x1,y1 = arr[0][:2].astype(float); x2,y2 = arr[1][:2].astype(float)
                return int(x1), int(y1), int(x2), int(y2)
            if arr.ndim == 2 and arr.shape[1] >=2:
                xs = arr[:,0].astype(float); ys = arr[:,1].astype(float)
                return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    except Exception:
        pass
    try:
        b = bbox
        if hasattr(b, "upper_left") and hasattr(b, "lower_right"):
            ul = b.upper_left; lr = b.lower_right
            def pt_xy(pt):
                if hasattr(pt, "x") and hasattr(pt, "y"):
                    return float(pt.x), float(pt.y)
                arr = np.array(pt).ravel()
                return float(arr[0]), float(arr[1])
            x1,y1 = pt_xy(ul); x2,y2 = pt_xy(lr)
            return int(x1), int(y1), int(x2), int(y2)
        if hasattr(b, "tl") and hasattr(b, "br"):
            tx,ty = getattr(b,"tl"); bx,by = getattr(b,"br")
            return int(tx), int(ty), int(bx), int(by)
        if hasattr(b, "xyxy"):
            arr = np.array(b.xyxy).ravel()
            if arr.size >= 4:
                return int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3])
        keys = ("x1","y1","x2","y2")
        if all(hasattr(b, k) for k in keys):
            return int(getattr(b,"x1")), int(getattr(b,"y1")), int(getattr(b,"x2")), int(getattr(b,"y2"))
    except Exception:
        pass
    try:
        s = str(bbox)
        pts = re.findall(r"Point\s*\(\s*x\s*=\s*([\-0-9.]+)\s*,\s*y\s*=\s*([\-0-9.]+)\s*\)", s)
        if len(pts) >= 2:
            x1,y1 = map(float, pts[0]); x2,y2 = map(float, pts[1])
            return int(x1), int(y1), int(x2), int(y2)
        floats = re.findall(r"[-+]?\d*\.\d+|\d+", s)
        if len(floats) >= 4:
            a = list(map(float, floats[:4]))
            if a[2] > a[0] and a[3] > a[1]:
                return int(a[0]), int(a[1]), int(a[2]), int(a[3])
            else:
                return int(a[0]), int(a[1]), int(a[0]+a[2]), int(a[1]+a[3])
    except Exception:
        pass
    raise ValueError(f"Unsupported bbox format: {type(bbox)} / repr: {repr(bbox)[:200]}")


def draw_annotations_pil(pil_img, detections):
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for d in detections:
        x1,y1,x2,y2 = map(int, d['xyxy'])
        name = d.get('name','Unknown')
        score = d.get('score', None)
        color = (0,200,0) if name != "Unknown" else (200,0,0)
        draw.rectangle([(x1,y1),(x2,y2)], outline=color, width=2)
        txt = name if score is None else f"{name} {score:.2f}"
        try:
            bb = draw.textbbox((0,0), txt, font=font)
            tw, th = bb[2]-bb[0], bb[3]-bb[1]
        except Exception:
            tw, th = font.getsize(txt)
        draw.rectangle([(x1, y1 - th - 4), (x1 + tw + 6, y1)], fill=color)
        draw.text((x1+3, y1-th-2), txt, fill=(0,0,0), font=font)
    return img


def draw_annotations_and_overlay(original_bgr, results_info, absent_list):
    pil = Image.fromarray(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for d in results_info:
        x1,y1,x2,y2 = map(int, d['xyxy'])
        name = d.get('name','Unknown')
        score = d.get('score', None)
        color = (0,200,0) if name != "Unknown" else (200,0,0)
        draw.rectangle([(x1,y1),(x2,y2)], outline=color, width=2)
        txt = name if score is None else f"{name} {score:.2f}"
        try:
            bb = draw.textbbox((0,0), txt, font=font)
            tw, th = bb[2]-bb[0], bb[3]-bb[1]
        except Exception:
            tw, th = font.getsize(txt)
        draw.rectangle([(x1, y1 - th - 4), (x1 + tw + 6, y1)], fill=color)
        draw.text((x1+3, y1-th-2), txt, fill=(0,0,0), font=font)
    if len(absent_list) > 0:
        text_lines = ["Absent:"]
        line = ""
        for nm in absent_list:
            if len(line) + len(nm) + 2 > 40:
                text_lines.append(line.strip(", "))
                line = nm + ", "
            else:
                line += nm + ", "
        if line:
            text_lines.append(line.strip(", "))
        annotated_rgba = pil.convert("RGBA")
        overlay = Image.new("RGBA", annotated_rgba.size, (255,255,255,0))
        odraw = ImageDraw.Draw(overlay)
        try:
            font2 = ImageFont.truetype("DejaVuSans.ttf", 16)
        except Exception:
            font2 = ImageFont.load_default()
        maxw = 0; totalh = 0
        for ln in text_lines:
            try:
                bb = odraw.textbbox((0,0), ln, font=font2)
                tw = bb[2]-bb[0]; th = bb[3]-bb[1]
            except Exception:
                tw, th = font2.getsize(ln)
            if tw > maxw: maxw = tw
            totalh += th + 4
        padding = 8
        box_w = maxw + padding*2
        box_h = totalh + padding*2
        odraw.rectangle([(6,6),(6+box_w,6+box_h)], fill=(255,255,255,210))
        y0 = 6 + padding
        for ln in text_lines:
            odraw.text((6+padding, y0), ln, fill=(180,0,0), font=font2)
            try:
                bb = odraw.textbbox((0,0), ln, font=font2); th = bb[3]-bb[1]
            except Exception:
                _, th = font2.getsize(ln)
            y0 += th + 4
        annotated_rgba = Image.alpha_composite(annotated_rgba, overlay)
        pil = annotated_rgba.convert("RGB")
    return pil


def make_thumbnail_from_bgr(cv2_img_bgr, size=128):
    if cv2_img_bgr is None or cv2_img_bgr.size == 0:
        return None
    pil = Image.fromarray(cv2.cvtColor(cv2_img_bgr, cv2.COLOR_BGR2RGB))
    pil.thumbnail((size,size))
    return pil


# ---------- Enrollment UI ----------
st.header("1) Enrollment")
col1, col2 = st.columns([1,2])
with col1:
    enroll_zip = st.file_uploader("Upload a ZIP of student folders (optional)", type=["zip"])
    st.markdown("**Or enroll a single student**")
    student_name = st.text_input("Student Name (required for single-student enroll)")
    student_sapid = st.text_input("SAP ID (required for single-student enroll)")
    student_email = st.text_input("Email (optional)")
    student_files = st.file_uploader("Upload images for the student (select multiple)", accept_multiple_files=True, type=["jpg","jpeg","png"])
    enroll_btn = st.button("Save student images")
with col2:
    st.subheader("Enrolled students")
    if DB_PATH.exists():
        db_show = robust_load_enroll_db(DB_PATH)
        enrolled_names = db_show["names"]
        df_show = pd.DataFrame({
            "name": db_show["names"],
            "sapid": db_show["sapids"],
            "email": db_show["emails"]
        })
        st.write(f"Enrolled: {len(enrolled_names)} students")
        st.dataframe(df_show, use_container_width=True)

        raw = np.load(DB_PATH, allow_pickle=True)
        missing = []
        if "sapids" not in set(raw.files):
            missing.append("sapids")
        if "emails" not in set(raw.files):
            missing.append("emails")
        if len(missing) > 0:
            if st.button(f"Add missing DB keys ({', '.join(missing)})"):
                try:
                    migrate_enroll_db_add_missing_keys(DB_PATH)
                    st.success("DB migrated. Reload the app to see updated fields.")
                except Exception as e:
                    st.error("Migration failed: " + str(e))
    else:
        st.info("No enrollment DB found yet.")


# handle zip
if enroll_zip is not None:
    tmp = tempfile.mkdtemp()
    try:
        z = zipfile.ZipFile(enroll_zip)
        z.extractall(tmp)
        moved = 0
        for entry in Path(tmp).iterdir():
            if entry.is_dir():
                dest = ENROLL_DIR / entry.name
                if dest.exists():
                    for p in entry.rglob("*"):
                        if p.is_file():
                            shutil.copy(p, dest / p.name)
                else:
                    shutil.move(str(entry), str(dest))
                moved += 1
        st.success(f"Extracted {moved} student folder(s) into enroll directory.")
    except Exception as e:
        st.error("Failed to extract zip: " + str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# handle single-student enroll
if enroll_btn:
    if student_name.strip() == "" or student_sapid.strip() == "" or len(student_files) == 0:
        st.warning("For single-student enrollment provide Student Name, SAP ID and at least one image.")
    else:
        safe_name = "".join(c for c in student_name if c.isalnum() or c in (" ", "_","-")).strip().replace(" ", "_")
        folder_name = f"{student_sapid.strip()}_{safe_name}"
        dst = ENROLL_DIR / folder_name
        dst.mkdir(parents=True, exist_ok=True)
        saved = 0
        for uf in student_files:
            try:
                data = uf.read()
                p = dst / uf.name
                with open(p, "wb") as f:
                    f.write(data)
                saved += 1
            except Exception as e:
                st.error("Failed to save " + uf.name + ": " + str(e))
        meta = {"name": student_name.strip(), "sapid": student_sapid.strip(), "email": student_email.strip()}
        try:
            with open(dst / "meta.json", "w", encoding="utf-8") as mf:
                json.dump(meta, mf, ensure_ascii=False, indent=2)
        except Exception:
            pass
        st.success(f"Saved {saved} image(s) for student '{student_name.strip()}' to {dst}")


# ---------- Create Enrollment DB ----------
st.markdown("---")
st.subheader("2) Create / Update Enrollment DB")
min_imgs = st.number_input("Min images per student (skip if fewer)", min_value=1, max_value=50, value=1)
create_db_btn = st.button("Create Enrollment DB from folder")

if create_db_btn:
    if not SCRFD_PATH.exists():
        st.error(f"SCRFD ONNX model not found at {SCRFD_PATH}. Upload one in the sidebar.")
    else:
        detector, threshold, app = init_models_cached(str(SCRFD_PATH))
        names = []; sapids = []; emails = []; embs = []
        student_dirs = [p for p in sorted(ENROLL_DIR.iterdir()) if p.is_dir()]
        if len(student_dirs) == 0:
            st.warning("No student folders found in enroll directory.")
        else:
            prog = st.progress(0)
            total = len(student_dirs)
            for i, sd in enumerate(student_dirs, start=1):
                prog.progress(int(i/total*100))
                meta = {"name": sd.name, "sapid":"", "email":""}
                try:
                    meta_file = sd / "meta.json"
                    if meta_file.exists():
                        with open(meta_file, "r", encoding="utf-8") as mf:
                            m = json.load(mf)
                            meta["name"] = m.get("name", meta["name"])
                            meta["sapid"] = m.get("sapid", meta["sapid"])
                            meta["email"] = m.get("email", meta["email"])
                    else:
                        # FIX: if folder name is purely numeric, treat it as SAPID (not as name)
                        parts = sd.name.split("_", 1)
                        if len(parts) == 2 and parts[0].isdigit():
                            meta["sapid"] = parts[0]
                            meta["name"] = parts[1].replace("_"," ")
                        else:
                            # if entire folder name is digits -> sapid only
                            if sd.name.isdigit():
                                meta["sapid"] = sd.name
                                meta["name"] = ""
                            else:
                                # otherwise treat folder name as student name (fallback)
                                meta["name"] = sd.name.replace("_"," ")
                except Exception:
                    pass

                imgs = sorted([p for p in sd.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}])
                if len(imgs) < min_imgs:
                    st.info(f"Skipping {meta['name'] or meta['sapid']}: not enough images ({len(imgs)})")
                    continue
                templates = []
                for p in imgs:
                    img = cv2.imread(str(p))
                    if img is None:
                        continue
                    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    dets = detector.detect(pil, threshold=threshold)
                    if len(dets) == 0:
                        continue
                    best = max(dets, key=lambda f: getattr(f,"probability",0.0))
                    try:
                        x1,y1,x2,y2 = _extract_xyxy_from_bbox(best.bbox)
                    except Exception:
                        continue
                    h,w = img.shape[:2]
                    pad = int(PAD_RATIO * max(1, (y2-y1)))
                    x1p,y1p = max(0, x1-pad), max(0, y1-pad)
                    x2p,y2p = min(w-1, x2+pad), min(h-1, y2+pad)
                    crop = img[y1p:y2p, x1p:x2p]
                    faces = app.get(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    if len(faces) == 0:
                        continue
                    face = faces[0]
                    if hasattr(face, "normed_embedding") and face.normed_embedding is not None:
                        emb = np.array(face.normed_embedding, dtype=np.float32)
                    elif hasattr(face, "embedding") and face.embedding is not None:
                        emb = np.array(face.embedding, dtype=np.float32); emb = emb / (np.linalg.norm(emb)+1e-10)
                    else:
                        continue
                    templates.append(emb)
                if len(templates) == 0:
                    st.write(f"No valid embeddings for {meta['name'] or meta['sapid']}; skipping.")
                    continue
                avg = np.mean(np.vstack(templates), axis=0); avg = avg / (np.linalg.norm(avg)+1e-10)
                names.append(meta["name"])
                sapids.append(meta["sapid"])
                emails.append(meta["email"])
                embs.append(avg)
                st.write(f"Enrolled {meta['name'] or meta['sapid']} (images used: {len(templates)})")
            prog.empty()
            if len(embs) == 0:
                st.error("No students enrolled (no valid embeddings).")
            else:
                embs = np.vstack(embs)
                np.savez_compressed(DB_PATH,
                                    names=np.array(names, dtype=object),
                                    sapids=np.array(sapids, dtype=object),
                                    emails=np.array(emails, dtype=object),
                                    embeddings=embs)
                st.success(f"Saved enrollment DB with {len(names)} students -> {DB_PATH}")


# ---------- Attendance UI ----------
st.markdown("---")
st.header("3) Mark Attendance (upload group photo)")
group_file = st.file_uploader("Upload group photo (single image) to mark attendance", type=["jpg","jpeg","png"])


def init_session_attendance():
    if 'attendance_results' not in st.session_state:
        st.session_state['attendance_results'] = None


init_session_attendance()

if st.button("Run Attendance") and group_file is not None:
    if not DB_PATH.exists():
        st.error("No enrollment DB found. Create it first.")
    elif not SCRFD_PATH.exists():
        st.error("SCRFD ONNX missing in models dir or upload via sidebar.")
    else:
        detector, threshold, app = init_models_cached(str(SCRFD_PATH))
        db = robust_load_enroll_db(DB_PATH)
        names = db["names"]; sapids = db["sapids"]; emails = db["emails"]; emb_db = db["embeddings"]
        if emb_db is None or emb_db.shape[0] == 0:
            st.error("Enrollment DB has no embeddings. Recreate DB.")
        else:
            group_bytes = group_file.read()
            group_img = cv2.imdecode(np.frombuffer(group_bytes, np.uint8), cv2.IMREAD_COLOR)
            if group_img is None:
                st.error("Failed to read uploaded image.")
            else:
                h,w = group_img.shape[:2]
                pil = Image.fromarray(cv2.cvtColor(group_img, cv2.COLOR_BGR2RGB))
                raw_dets = detector.detect(pil, threshold=threshold)
                detections = []
                for f in raw_dets:
                    try:
                        x1,y1,x2,y2 = _extract_xyxy_from_bbox(f.bbox)
                    except Exception:
                        continue
                    x1,y1 = max(0,x1), max(0,y1)
                    x2,y2 = min(w-1,x2), min(h-1,y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    detections.append({"xyxy":(x1,y1,x2,y2), "prob": float(getattr(f,"probability",0.0))})
                results_info = []
                present_indices = set()
                face_previews = []
                for det in detections:
                    x1,y1,x2,y2 = det['xyxy']
                    crop = group_img[y1:y2, x1:x2].copy()
                    faces = app.get(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    if len(faces) == 0:
                        results_info.append({"xyxy":det["xyxy"], "name":"Unknown", "score":0.0})
                        face_previews.append({"thumb":make_thumbnail_from_bgr(crop), "name":"Unknown", "score":0.0})
                        continue
                    face = faces[0]
                    if hasattr(face, "normed_embedding") and face.normed_embedding is not None:
                        emb = np.array(face.normed_embedding, dtype=np.float32)
                    elif hasattr(face, "embedding") and face.embedding is not None:
                        emb = np.array(face.embedding, dtype=np.float32); emb = emb / (np.linalg.norm(emb)+1e-10)
                    else:
                        results_info.append({"xyxy":det["xyxy"], "name":"Unknown", "score":0.0})
                        face_previews.append({"thumb":make_thumbnail_from_bgr(crop), "name":"Unknown", "score":0.0})
                        continue
                    sims = cosine_similarity(emb.reshape(1,-1), emb_db).ravel()
                    best_idx = int(np.argmax(sims)); best_sim = float(sims[best_idx])
                    if best_sim >= SIM_THRESHOLD:
                        present_indices.add(best_idx)
                        name = names[best_idx] or sapids[best_idx] or "Unknown"
                    else:
                        name = "Unknown"
                    results_info.append({"xyxy":det["xyxy"], "name":name, "score":best_sim})
                    face_previews.append({"thumb":make_thumbnail_from_bgr(crop), "name":name, "score":best_sim})

                enrolled_indices = set(range(len(names)))
                absent_indices = sorted(list(enrolled_indices - present_indices))
                absent_names = [names[i] or sapids[i] or "" for i in absent_indices]
                annotated_pil = draw_annotations_and_overlay(group_img, results_info, absent_names)
                buf = BytesIO(); annotated_pil.save(buf, format="JPEG")
                st.session_state['attendance_results'] = {
                    "names": names,
                    "sapids": sapids,
                    "emails": emails,
                    "emb_db": emb_db,
                    "group_bgr": group_img,
                    "results_info": results_info,
                    "present_indices": list(present_indices),
                    "absent_indices": absent_indices,
                    "face_previews": face_previews,
                    "annotated_bytes": buf.getvalue()
                }
                st.success(f"Attendance done. Present: {len(present_indices)}; Absent: {len(absent_indices)}")


# ---------- Render attendance results & manual verify ----------
res = st.session_state.get('attendance_results', None)
if res is not None:
    names = res["names"]; sapids = res["sapids"]; emails = res["emails"]
    results_info = res["results_info"]
    present_indices = set(res["present_indices"])
    absent_indices = res["absent_indices"]
    face_previews = res["face_previews"]
    annotated_bytes = res.get("annotated_bytes", None)
    group_bgr = res["group_bgr"]

    st.subheader("Annotated result")
    if annotated_bytes is not None:
        st.image(Image.open(BytesIO(annotated_bytes)), use_container_width=True)

    status_list = ["Present" if i in present_indices else "Absent" for i in range(len(names))]
    df_roster = pd.DataFrame({
        "name": names,
        "sapid": sapids,
        "email": emails,
        "status": status_list
    })

    def _highlight_row(row):
        color = "#000000" if row["status"] == "Present" else "#878282"
        return [f"background-color: {color}"] * len(row)

    st.subheader("Class roster — Present / Absent")
    try:
        styled = df_roster.style.apply(_highlight_row, axis=1)
        st.dataframe(styled, use_container_width=True)
    except Exception:
        st.dataframe(df_roster, use_container_width=True)

    st.subheader("Detected faces (name — similarity)")
    if len(face_previews) == 0:
        st.info("No faces detected.")
    else:
        cols_per_row = 6
        rows = [face_previews[i:i+cols_per_row] for i in range(0, len(face_previews), cols_per_row)]
        for row in rows:
            cols = st.columns(len(row))
            for col, item in zip(cols, row):
                with col:
                    if item["thumb"] is not None:
                        st.image(item["thumb"], use_container_width=True)
                    else:
                        st.image(Image.new("RGB",(128,128),(200,200,200)), use_container_width=True)
                    st.caption(f"{item['name']} — {item['score']:.3f}")

    st.markdown("---")
    st.subheader("Manual verification for absent students")
    if len(absent_indices) == 0:
        st.success("No absent students — all enrolled detected.")
    else:
        absent_display = [f"{names[i] or sapids[i]} ({sapids[i]})" for i in absent_indices]
        st.info(f"{len(absent_indices)} absent. Select one to inspect and optionally mark present.")
        selected_display = st.selectbox("Select absent student", options=absent_display, key="absent_select")
        if selected_display:
            sel_index = absent_display.index(selected_display)
            sel_idx = absent_indices[sel_index]
            sel_name = names[sel_idx] or ""
            sel_sapid = sapids[sel_idx] or ""
            sel_email = emails[sel_idx] or ""
            st.write(f"**{sel_name or sel_sapid}** — SAPID: `{sel_sapid}` — Email: `{sel_email}`")
            sdir = None
            guess_folder = f"{sel_sapid}_{(sel_name.replace(' ','_') if sel_name else '')}".rstrip("_")
            if sel_sapid and (ENROLL_DIR / guess_folder).exists():
                sdir = ENROLL_DIR / guess_folder
            else:
                for p in ENROLL_DIR.iterdir():
                    if not p.is_dir(): continue
                    meta_file = p / "meta.json"
                    if meta_file.exists():
                        try:
                            m = json.load(open(meta_file,'r',encoding='utf-8'))
                            if (sel_sapid and m.get("sapid","")==sel_sapid) or (sel_name and m.get("name","")==sel_name):
                                sdir = p; break
                        except Exception:
                            pass
                    if sel_sapid and sel_sapid in p.name:
                        sdir = p; break
                    if sel_name and sel_name.replace(" ","_") in p.name:
                        sdir = p; break
            imgs = sorted([p for p in (sdir.iterdir() if sdir and sdir.exists() else []) if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}]) if sdir else []
            if len(imgs) == 0:
                st.warning("No enrolled images found for this student (folder missing or empty).")
            else:
                st.write(f"Showing {len(imgs)} enrolled image(s) for **{sel_name or sel_sapid}**")
                cols = st.columns(min(len(imgs), 6))
                for i,p in enumerate(imgs):
                    with cols[i % len(cols)]:
                        try:
                            im = Image.open(p).convert("RGB")
                            st.image(im, use_container_width=True)
                        except Exception:
                            st.image(Image.new("RGB",(128,128),(200,200,200)), use_container_width=True)
                        st.caption(p.name)
                if st.button(f"Mark '{sel_name or sel_sapid}' as Present"):
                    present_indices.add(sel_idx)
                    if sel_idx in absent_indices:
                        absent_indices.remove(sel_idx)
                    absent_names_now = [names[i] or sapids[i] for i in absent_indices]
                    new_annot = draw_annotations_and_overlay(group_bgr, results_info, absent_names_now)
                    buf = BytesIO(); new_annot.save(buf, format="JPEG")
                    res["annotated_bytes"] = buf.getvalue()
                    res["present_indices"] = list(present_indices)
                    res["absent_indices"] = absent_indices
                    st.session_state['attendance_results'] = res
                    st.success(f"Marked {sel_name or sel_sapid} as Present. Roster and annotated image updated.")

    st.markdown("---")
    st.subheader("Downloads")
    attendance = []
    present_set = set(res["present_indices"])
    for idx in range(len(names)):
        attendance.append({
            "name": names[idx],
            "sapid": sapids[idx],
            "email": emails[idx],
            "status": "Present" if idx in present_set else "Absent"
        })
    df_att = pd.DataFrame(attendance)
    st.download_button("Download attendance CSV", data=df_att.to_csv(index=False).encode("utf-8"), file_name="attendance.csv", mime="text/csv")
    ann_bytes = res.get("annotated_bytes", None)
    if ann_bytes is not None:
        st.download_button("Download annotated image", data=ann_bytes, file_name="annotated.jpg", mime="image/jpeg")


# ---------- Utilities ----------
st.markdown("---")
st.header("Utilities")
if DB_PATH.exists():
    with open(DB_PATH, "rb") as f:
        dbb = f.read()
    st.download_button("Download enrollment DB (.npz)", data=dbb, file_name="enroll_db.npz")
else:
    st.info("No enrollment DB available.")

st.caption("Tips: tune similarity threshold. For production, align faces and use ONNX runtime optimizations or GPU.")
