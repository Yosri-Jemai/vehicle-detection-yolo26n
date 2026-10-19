"""
Real-time vehicle detection on uploaded videos.
Model : Yosriii/vehicle-detection-yolo26n (Hugging Face)
Run   : uvicorn app:app --host 0.0.0.0 --port 8000
"""
import os
import time
import uuid
from collections import Counter
from pathlib import Path

import cv2
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# ---------------------------------------------------------------- config ---
load_dotenv()  # reads HF_TOKEN from the .env file

REPO_ID = "Yosriii/vehicle-detection-yolo26n"
WEIGHTS_FILE = "best.pt"
HF_TOKEN = os.getenv("HF_TOKEN")  # needed because the repo is private
CONF = 0.25      # confidence threshold
IMGSZ = 640      # same size as training
JPEG_QUALITY = 80
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Download the weights once at startup (cached afterwards)
WEIGHTS_PATH = hf_hub_download(REPO_ID, WEIGHTS_FILE, token=HF_TOKEN)

app = FastAPI(title="Vehicle Detection")
JOBS: dict[str, dict] = {}  # job_id -> state


# ------------------------------------------------------------- streaming ---
def frame_generator(job_id: str):
    """Reads the video, runs YOLO + tracking frame by frame, yields MJPEG."""
    job = JOBS[job_id]
    # One model instance per stream: the tracker keeps state inside it
    model = YOLO(WEIGHTS_PATH)

    cap = cv2.VideoCapture(job["path"])
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    job["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_time = 1.0 / src_fps

    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                break

            result = model.track(
                frame,
                persist=True,
                conf=CONF,
                imgsz=IMGSZ,
                tracker="bytetrack.yaml",
                verbose=False,
            )[0]

            live = Counter()
            if result.boxes is not None and len(result.boxes):
                classes = result.boxes.cls.int().tolist()
                ids = (
                    result.boxes.id.int().tolist()
                    if result.boxes.id is not None
                    else [None] * len(classes)
                )
                for tid, c in zip(ids, classes):
                    name = model.names[c]
                    live[name] += 1
                    if tid is not None:
                        job["tracks"][tid] = name  # unique vehicle -> class

            job["live"] = dict(live)
            job["frame"] += 1

            annotated = result.plot()
            ok, buf = cv2.imencode(
                ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            if not ok:
                continue
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + buf.tobytes()
                + b"\r\n"
            )

            # Keep the playback close to real time if inference is faster
            elapsed = time.time() - t0
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
    finally:
        cap.release()
        job["done"] = True


# ---------------------------------------------------------------- routes ---
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("video/"):
        raise HTTPException(400, "Please upload a video file.")
    job_id = uuid.uuid4().hex[:10]
    path = UPLOAD_DIR / f"{job_id}{Path(file.filename or 'v.mp4').suffix or '.mp4'}"
    with open(path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)
    JOBS[job_id] = {
        "path": str(path),
        "frame": 0,
        "total_frames": 0,
        "tracks": {},
        "live": {},
        "done": False,
    }
    return {"job_id": job_id}


@app.get("/stream/{job_id}")
def stream(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Unknown job.")
    return StreamingResponse(
        frame_generator(job_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/stats/{job_id}")
def stats(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    totals = Counter(job["tracks"].values())
    return {
        "frame": job["frame"],
        "total_frames": job["total_frames"],
        "done": job["done"],
        "live": job["live"],                       # vehicles in current frame
        "unique": dict(totals),                    # distinct vehicles so far
        "unique_total": sum(totals.values()),
    }


# ------------------------------------------------------------------ page ---
PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Vehicle Detection</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root{--bg:#f4f6f8;--panel:#fff;--border:#d9dee5;--text:#1f2937;--muted:#6b7280;--accent:#1d4ed8}
  *{box-sizing:border-box}
  body{font-family:"Segoe UI",Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--text);margin:0}
  header{background:var(--panel);border-bottom:1px solid var(--border);padding:16px 32px}
  header h1{margin:0;font-size:18px;font-weight:600;letter-spacing:.2px}
  main{padding:24px 32px;max-width:1280px;margin:0 auto}
  .panel{background:var(--panel);border:1px solid var(--border);border-radius:2px;padding:16px}
  .toolbar{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
  .toolbar input[type=file]{font-size:14px;color:var(--text)}
  button{background:var(--accent);color:#fff;border:1px solid var(--accent);border-radius:2px;padding:8px 20px;font-size:14px;font-weight:500;cursor:pointer}
  button:hover{background:#1e40af}
  button:disabled{opacity:.5;cursor:default}
  #msg{color:var(--muted);font-size:14px}
  .bar{height:4px;background:#e5e7eb;margin-top:14px}
  .bar div{height:100%;width:0;background:var(--accent)}
  .row{display:flex;gap:24px;flex-wrap:wrap;margin-top:24px;align-items:flex-start}
  .video{flex:1 1 640px}
  #video{display:block;width:100%;background:#111827;min-height:320px;border:1px solid var(--border)}
  .side{flex:0 0 280px}
  h3{margin:0 0 12px;font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:.6px;color:var(--muted)}
  table{border-collapse:collapse;width:100%}
  td{padding:9px 4px;border-bottom:1px solid var(--border);font-size:15px}
  td:last-child{text-align:right;font-weight:600;font-variant-numeric:tabular-nums}
  td.empty{color:var(--muted);font-weight:400;text-align:left}
</style>
</head>
<body>
<header><h1>Vehicle Detection</h1></header>
<main>
  <div class="panel">
    <div class="toolbar">
      <input type="file" id="file" accept="video/*">
      <button id="go">Detect</button>
      <span id="msg"></span>
    </div>
    <div class="bar"><div id="progress"></div></div>
  </div>
  <div class="row">
    <div class="panel video"><img id="video" alt=""></div>
    <div class="panel side">
      <h3>Vehicles in frame</h3>
      <table id="live"></table>
    </div>
  </div>
</main>
<script>
let timer;
const $ = id => document.getElementById(id);
const fill = (el, obj) => el.innerHTML = Object.keys(obj).length
  ? Object.entries(obj).map(([k,v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('')
  : '<tr><td class="empty">No detections</td></tr>';
fill($('live'), {});

$('go').onclick = async () => {
  const f = $('file').files[0];
  if (!f) return alert('Choose a video first.');
  $('go').disabled = true; $('msg').textContent = 'Uploading...';
  const fd = new FormData(); fd.append('file', f);
  const res = await fetch('/upload', {method:'POST', body: fd});
  if (!res.ok) { $('msg').textContent = 'Upload failed'; $('go').disabled = false; return; }
  const {job_id} = await res.json();
  $('msg').textContent = 'Processing...';
  $('video').src = '/stream/' + job_id;
  clearInterval(timer);
  timer = setInterval(async () => {
    const s = await (await fetch('/stats/' + job_id)).json();
    fill($('live'), s.live);
    if (s.total_frames) $('progress').style.width = (100 * s.frame / s.total_frames) + '%';
    if (s.done) { clearInterval(timer); $('msg').textContent = 'Done'; $('go').disabled = false; }
  }, 500);
};
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE
