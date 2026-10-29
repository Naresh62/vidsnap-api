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
    render_secret = "/etc/secrets/cookies.txt"
    writable_cookie = "/tmp/yt_cookies.txt"
    if os.path.exists(render_secret):
        try:
            shutil.copyfile(render_secret, writable_cookie)
            return writable_cookie
        except Exception:
            pass
    elif os.path.exists("cookies.txt"):
        try:
            shutil.copyfile("cookies.txt", writable_cookie)
            return writable_cookie
        except Exception:
            pass
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
        "tv_engine": "active",
        "cookies_loaded": cookie_path is not None
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
    temp_video_prefix = f"video_{session_id}"
    slides_dir = f"slides_{session_id}"
    output_pdf = f"Lecture_Slides_{session_id}.pdf"
    temp_video_file = None

    cookie_file = get_writable_cookie_file()

    # Smart TV & TV Embedded bypass (Datacenter IP par 100% bypass karta hai)
    ydl_opts = {
        'format': 'bestvideo[height<=480]/best[height<=480]/worstvideo/worst/best',
        'outtmpl': f"{temp_video_prefix}.%(ext)s",
        'quiet': True,
        'no_warnings': True,
        'extractor_args': {
            'youtube': {
                'player_client': ['tv_embedded', 'tv', 'android_vr'],
                'player_skip': ['webpage', 'configs']
            }
        },
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (SMART-TV; Linux; Tizen 6.0) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/4.0 Chrome/76.0.3809.146 TV Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        }
    }

    if cookie_file:
        ydl_opts['cookiefile'] = cookie_file

    try:
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([clean_url])
        except Exception:
            # Fallback: Agar pehla fail ho, toh bina cookies ke clean iOS client try karein
            fallback_opts = {
                'format': 'bestvideo[height<=480]/best[height<=480]/worstvideo/worst/best',
                'outtmpl': f"{temp_video_prefix}.%(ext)s",
                'quiet': True,
                'no_warnings': True,
                'extractor_args': {
                    'youtube': {
                        'player_client': ['ios', 'android_vr']
                    }
                }
            }
            with yt_dlp.YoutubeDL(fallback_opts) as ydl:
                ydl.download([clean_url])

        matching_files = glob.glob(f"{temp_video_prefix}.*")
        if not matching_files:
            raise HTTPException(status_code=500, detail="Video stream capture nahi ho saki.")
        
        temp_video_file = matching_files[0]

        # OpenCV Frame Capture
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

        if temp_video_file and os.path.exists(temp_video_file):
            os.remove(temp_video_file)

        if not saved_slides:
            cleanup_files(slides_dir)
            raise HTTPException(status_code=400, detail="Koi distinct slide detect nahi hui.")

        # PDF Compilation
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
