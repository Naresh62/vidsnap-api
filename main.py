import os
import re
import glob
import uuid
import shutil
import cv2
import numpy as np
import img2pdf
import requests
import yt_dlp
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="VidSnapHub Slide Extractor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoRequest(BaseModel):
    url: str

def cleanup_files(*paths):
    for path in paths:
        if not path:
            continue
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    os.remove(path)
                except Exception:
                    pass

def fetch_stream_via_proxy_resolvers(video_id: str, output_path: str) -> bool:
    """YouTube bot detection ko bypass karke direct video stream download karta hai"""
    
    # 1. Invidious API Network (Datacenter-friendly nodes)
    invidious_nodes = [
        f"https://inv.nadeko.net/api/v1/videos/{video_id}",
        f"https://invidious.nerdvpn.de/api/v1/videos/{video_id}",
        f"https://inv.tux.pizza/api/v1/videos/{video_id}",
        f"https://invidious.jing.rocks/api/v1/videos/{video_id}"
    ]

    for node_url in invidious_nodes:
        try:
            res = requests.get(node_url, timeout=6)
            if res.status_code == 200:
                data = res.json()
                streams = data.get("formatStreams") or []
                if not streams:
                    # video-only stream fallback
                    streams = [f for f in data.get("adaptiveFormats", []) if "video" in f.get("type", "")]

                if streams:
                    # Lowest resolution (360p/480p) chunein taaki speed tez ho
                    stream_url = streams[0].get("url")
                    if stream_url:
                        with requests.get(stream_url, stream=True, timeout=25) as r:
                            r.raise_for_status()
                            with open(output_path, 'wb') as f:
                                for chunk in r.iter_content(chunk_size=1024 * 1024):
                                    if chunk:
                                        f.write(chunk)
                        if os.path.exists(output_path) and os.path.getsize(output_path) > 500000:
                            return True
        except Exception:
            continue

    # 2. Piped API Network (Secondary fallback)
    piped_nodes = [
        f"https://pipedapi.kavin.rocks/streams/{video_id}",
        f"https://api.piped.privacydev.net/streams/{video_id}",
        f"https://pipedapi.leptons.xyz/streams/{video_id}"
    ]

    for node_url in piped_nodes:
        try:
            res = requests.get(node_url, timeout=6)
            if res.status_code == 200:
                data = res.json()
                streams = data.get("videoStreams", [])
                if streams:
                    stream_url = streams[0].get("url")
                    if stream_url:
                        with requests.get(stream_url, stream=True, timeout=25) as r:
                            r.raise_for_status()
                            with open(output_path, 'wb') as f:
                                for chunk in r.iter_content(chunk_size=1024 * 1024):
                                    if chunk:
                                        f.write(chunk)
                        if os.path.exists(output_path) and os.path.getsize(output_path) > 500000:
                            return True
        except Exception:
            continue

    return False

@app.get("/")
def home():
    return {
        "status": "running", 
        "service": "VidSnapHub Slide API", 
        "engine": "Auto-Stream-Resolver Active"
    }

@app.post("/extract")
async def extract_slides_endpoint(req: VideoRequest, background_tasks: BackgroundTasks):
    url = req.url.strip()
    
    match = re.search(r'(?:v=|\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})', url)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL format.")
    
    video_id = match.group(1)
    session_id = str(uuid.uuid4())[:8]
    temp_video = f"/tmp/video_{session_id}.mp4"
    slides_dir = f"/tmp/slides_{session_id}"
    output_pdf = f"/tmp/Lecture_Slides_{session_id}.pdf"

    try:
        # Step 1: Proxy Stream Resolver se video download karein (Zero bot detection)
        download_success = fetch_stream_via_proxy_resolvers(video_id, temp_video)

        # Step 2: Fallback to yt-dlp android_vr client if resolver fails
        if not download_success:
            ydl_opts = {
                'format': 'bestvideo[height<=480]/worstvideo/worst/best',
                'outtmpl': temp_video,
                'quiet': True,
                'no_warnings': True,
                'extractor_args': {
                    'youtube': {
                        'player_client': ['android_vr', 'tv_embedded']
                    }
                }
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

        if not os.path.exists(temp_video) or os.path.getsize(temp_video) < 100000:
            raise HTTPException(status_code=500, detail="Video stream capture nahi ho saki.")

        # Step 3: OpenCV Frame Capture (Slide Extraction)
        os.makedirs(slides_dir, exist_ok=True)
        cap = cv2.VideoCapture(temp_video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        interval = int(fps * 1.5)

        prev_gray = None
        saved_slides = []
        frame_idx = 0
        slide_num = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                if prev_gray is None:
                    slide_num += 1
                    path = f"{slides_dir}/slide_{slide_num:03d}.jpg"
                    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    saved_slides.append(path)
                    prev_gray = gray
                else:
                    diff = cv2.absdiff(prev_gray, gray)
                    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                    change = np.count_nonzero(thresh) / (gray.shape[0] * gray.shape[1])

                    # 18% change hone par new slide
                    if change > 0.18:
                        slide_num += 1
                        path = f"{slides_dir}/slide_{slide_num:03d}.jpg"
                        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        saved_slides.append(path)
                        prev_gray = gray

            frame_idx += 1
        cap.release()

        # Temporary video delete karein
        if os.path.exists(temp_video):
            os.remove(temp_video)

        if not saved_slides:
            cleanup_files(slides_dir)
            raise HTTPException(status_code=400, detail="Koi distinct slide detect nahi hui.")

        # Step 4: PDF Compilation
        with open(output_pdf, "wb") as f:
            f.write(img2pdf.convert(saved_slides))

        cleanup_files(slides_dir)
        background_tasks.add_task(cleanup_files, output_pdf)

        return FileResponse(
            output_pdf, 
            media_type='application/pdf', 
            filename=f"Lecture_Slides_{video_id}.pdf"
        )

    except Exception as e:
        cleanup_files(temp_video, slides_dir, output_pdf)
        raise HTTPException(status_code=500, detail=str(e))
