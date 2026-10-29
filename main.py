import os
import re
import glob
import uuid
import shutil
import cv2
import numpy as np
import img2pdf
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

def get_writable_cookie_file():
    """Render ki read-only secret file ko writable /tmp directory me copy karta hai"""
    render_secret = "/etc/secrets/cookies.txt"
    writable_cookie = "/tmp/yt_cookies.txt"
    
    if os.path.exists(render_secret):
        try:
            shutil.copyfile(render_secret, writable_cookie)
            return writable_cookie
        except Exception as e:
            print("Secret copy error:", e)
    elif os.path.exists("cookies.txt"):
        try:
            shutil.copyfile("cookies.txt", writable_cookie)
            return writable_cookie
        except Exception as e:
            print("Local cookie copy error:", e)
            
    return None

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

@app.get("/")
def home():
    cookie_path = get_writable_cookie_file()
    return {
        "status": "running", 
        "service": "VidSnapHub Slide API", 
        "cookies_ready": cookie_path is not None,
        "cookie_path": cookie_path
    }

@app.post("/extract")
async def extract_slides_endpoint(req: VideoRequest, background_tasks: BackgroundTasks):
    url = req.url.strip()
    
    match = re.search(r'(?:v=|\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})', url)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL format.")
    
    video_id = match.group(1)
    clean_url = f"https://www.youtube.com/watch?v={video_id}"
    
    session_id = str(uuid.uuid4())[:8]
    temp_video_prefix = f"/tmp/video_{session_id}"
    slides_dir = f"/tmp/slides_{session_id}"
    output_pdf = f"/tmp/Lecture_Slides_{session_id}.pdf"
    temp_video_file = None

    try:
        # Writable cookie path fetch karein
        cookie_file = get_writable_cookie_file()
        if not cookie_file:
            raise HTTPException(status_code=500, detail="Server par cookies.txt nahi mili.")

        # 1. yt-dlp Configuration (Cookies STRICTLY PASSED)
        ydl_opts = {
            'cookiefile': cookie_file,
            'format': 'bestvideo[height<=480]/best[height<=480]/worstvideo/worst/best',
            'outtmpl': f"{temp_video_prefix}.%(ext)s",
            'quiet': True,
            'no_warnings': True,
            'nocheckcertificate': True,
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'ios', 'web']
                }
            }
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([clean_url])

        # Downloaded file verify karein (chahe extension .mp4 ho ya .webm)
        matching_files = glob.glob(f"{temp_video_prefix}.*")
        if not matching_files:
            raise HTTPException(status_code=500, detail="Video stream download nahi ho saki.")
        
        temp_video_file = matching_files[0]

        # 2. OpenCV Slide Extraction
        os.makedirs(slides_dir, exist_ok=True)
        cap = cv2.VideoCapture(temp_video_file)
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

                    if change > 0.18:
                        slide_num += 1
                        path = f"{slides_dir}/slide_{slide_num:03d}.jpg"
                        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        saved_slides.append(path)
                        prev_gray = gray

            frame_idx += 1
        cap.release()

        # Temporary video file turant delete karein taaki storage bache
        if temp_video_file and os.path.exists(temp_video_file):
            os.remove(temp_video_file)

        if not saved_slides:
            cleanup_files(slides_dir)
            raise HTTPException(status_code=400, detail="Koi distinct slide detect nahi hui.")

        # 3. PDF Compilation
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
        for f in glob.glob(f"{temp_video_prefix}.*"):
            cleanup_files(f)
        cleanup_files(slides_dir, output_pdf)
        raise HTTPException(status_code=500, detail=str(e))
