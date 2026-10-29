import logging
import math
import os
import re
import shutil
import tempfile
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
import img2pdf
import numpy as np
import yt_dlp
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vidsnap")

app = FastAPI(title="VidSnapHub Slide Extractor API")

# Production mein CORS_ORIGINS mein apni website ka URL set karein.
origins = [
    value.strip()
    for value in os.getenv("CORS_ORIGINS", "*").split(",")
    if value.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

COOKIE_SOURCE = os.getenv(
    "YOUTUBE_COOKIES_FILE",
    "/etc/secrets/cookies.txt",
)

MAX_DURATION_SECONDS = 60 * 60
MAX_VIDEO_BYTES = 500 * 1024 * 1024
MAX_SLIDES = 400

SAMPLE_SECONDS = 1.5
CHANGE_THRESHOLD = 0.18


class VideoRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)


def cleanup_directory(directory):
    shutil.rmtree(directory, ignore_errors=True)


def extract_video_id(value):
    value = value.strip()

    # Raw video ID bhi accept karein.
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value

    if not value.startswith(("https://", "http://")):
        value = "https://" + value

    try:
        parsed = urlsplit(value)
        hostname = (parsed.hostname or "").lower()

        if parsed.username or parsed.password:
            raise ValueError("Invalid URL")

        if hostname in {"youtu.be", "www.youtu.be"}:
            video_id = parsed.path.strip("/")

        elif hostname in {
            "youtube.com",
            "www.youtube.com",
            "m.youtube.com",
            "music.youtube.com",
        }:
            if parsed.path.rstrip("/") == "/watch":
                video_id = parse_qs(parsed.query).get("v", [""])[0]
            else:
                parts = parsed.path.strip("/").split("/")
                if len(parts) != 2 or parts[0] not in {
                    "embed", "shorts", "live"
                }:
                    raise ValueError("Unsupported YouTube URL")

                video_id = parts[1]

        else:
            raise ValueError("Unsupported hostname")

        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            raise ValueError("Invalid video ID")

        return video_id

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Valid YouTube video link paste karein.",
        )


def prepare_cookie_file(workdir):
    source = Path(COOKIE_SOURCE)

    # Public videos ke liye cookies ke bina bhi attempt karein.
    if not source.is_file():
        return None

    destination = workdir / "cookies.txt"
    shutil.copyfile(source, destination)
    os.chmod(destination, 0o600)

    # Netscape cookie file header check.
    with destination.open("r", encoding="utf-8-sig") as handle:
        header = handle.readline().strip()

    if header not in {
        "# Netscape HTTP Cookie File",
        "# HTTP Cookie File",
    }:
        raise HTTPException(
            status_code=503,
            detail=(
                "Server cookies.txt ka format invalid hai. "
                "Netscape-format cookies file configure karein."
            ),
        )

    return str(destination)


def validate_video(info, *, incomplete):
    if info.get("is_live") or info.get("live_status") == "is_live":
        return "Live streams supported nahi hain."

    duration = info.get("duration")

    if duration is not None and duration > MAX_DURATION_SECONDS:
        return "Maximum 60-minute video supported hai."

    return None


def enforce_download_limit(progress):
    downloaded = progress.get("downloaded_bytes") or 0
    if downloaded > MAX_VIDEO_BYTES:
        raise yt_dlp.utils.DownloadError(
            "Video exceeds the download size limit."
        )


def download_video(clean_url, workdir, cookie_file):
    node_path = shutil.which("node")
    if not node_path:
        raise HTTPException(
            status_code=503,
            detail="Server par Node.js missing hai. Node 22+ configure karke redeploy karein.",
        )
    try:
        node_version = subprocess.run(
            [node_path, "--version"], check=True, capture_output=True,
            text=True, timeout=10,
        ).stdout.strip()
        if int(node_version.lstrip("v").split(".")[0]) < 22:
            raise ValueError("Node 22+ required")
    except (subprocess.SubprocessError, ValueError, OSError):
        raise HTTPException(
            status_code=503,
            detail="Server par working Node.js 22+ required hai.",
        )
    logger.info("YouTube JS runtime: Node %s", node_version)
    options = {
        "js_runtimes": {"node": {"path": node_path}},
        "format": "bestvideo[height<=720]/best[height<=720]",
        "outtmpl": str(workdir / "video.%(ext)s"),
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
        "socket_timeout": 30,
        "retries": 2,
        "fragment_retries": 2,
        "concurrent_fragment_downloads": 1,
        "max_filesize": MAX_VIDEO_BYTES,
        "match_filter": validate_video,
        "progress_hooks": [enforce_download_limit],
    }

    if cookie_file:
        options["cookiefile"] = cookie_file

    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(clean_url, download=True)

    if not info:
        raise HTTPException(
            status_code=400,
            detail="Video unavailable hai ya processing limit se bahar hai.",
        )

    downloaded_files = [
        path
        for path in workdir.glob("video.*")
        if path.is_file()
        and path.suffix.lower() not in {
            ".part", ".ytdl", ".json", ".temp"
        }
    ]

    if len(downloaded_files) != 1:
        raise HTTPException(
            status_code=502,
            detail=(
                "Video download complete nahi hui. "
                "Video availability aur server logs check karein."
            ),
        )

    video_path = downloaded_files[0]

    if video_path.stat().st_size > MAX_VIDEO_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Video size limit se badi hai.",
        )

    return video_path


def extract_slide_images(video_path, workdir):
    slides_directory = workdir / "slides"
    slides_directory.mkdir()

    capture = cv2.VideoCapture(str(video_path))
    saved_slides = []

    try:
        if not capture.isOpened():
            raise HTTPException(
                status_code=422,
                detail="Downloaded video ko decode nahi kiya ja saka.",
            )

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0

        interval = max(1, round(fps * SAMPLE_SECONDS))
        max_frames = int(fps * MAX_DURATION_SECONDS)

        previous_gray = None
        frame_index = 0

        while True:
            # Non-sampled frames ko retrieve/decode karne se bachein.
            if not capture.grab():
                break

            if frame_index >= max_frames:
                raise HTTPException(
                    status_code=413,
                    detail="Video maximum processing duration se badi hai.",
                )

            if frame_index % interval == 0:
                success, frame = capture.retrieve()

                if not success or frame is None:
                    raise HTTPException(
                        status_code=422,
                        detail="Video frame decode nahi ho saka.",
                    )

                # Comparison ko fixed resolution par karein.
                comparison = cv2.resize(frame, (640, 360))
                gray = cv2.cvtColor(
                    comparison,
                    cv2.COLOR_BGR2GRAY,
                )
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                save_frame = previous_gray is None

                if previous_gray is not None:
                    difference = cv2.absdiff(previous_gray, gray)

                    _, threshold = cv2.threshold(
                        difference,
                        25,
                        255,
                        cv2.THRESH_BINARY,
                    )

                    change = (
                        np.count_nonzero(threshold)
                        / threshold.size
                    )
                    save_frame = change > CHANGE_THRESHOLD

                if save_frame:
                    if len(saved_slides) >= MAX_SLIDES:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                "400-slide limit exceed hui. "
                                "Chhoti video use karein."
                            ),
                        )

                    slide_path = slides_directory / (
                        f"slide_{len(saved_slides) + 1:04d}.jpg"
                    )

                    written = cv2.imwrite(
                        str(slide_path),
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 92],
                    )

                    if not written:
                        raise RuntimeError("Could not save slide image")

                    saved_slides.append(str(slide_path))
                    previous_gray = gray

            frame_index += 1

    finally:
        capture.release()

    if not saved_slides:
        raise HTTPException(
            status_code=422,
            detail="Video se koi frame extract nahi hua.",
        )

    return saved_slides


@app.get("/")
def home():
    return {
        "status": "running",
        "service": "VidSnapHub Slide Extractor API",
        # File presence does not confirm cookie validity.
        "cookies_file_present": Path(COOKIE_SOURCE).is_file(),
    }


@app.post("/extract")
def extract_slides_endpoint(
    request: VideoRequest,
    background_tasks: BackgroundTasks,
):
    video_id = extract_video_id(request.url)
    clean_url = f"https://www.youtube.com/watch?v={video_id}"

    workdir = Path(tempfile.mkdtemp(prefix="vidsnap_"))
    response_ready = False

    try:
        cookie_file = prepare_cookie_file(workdir)

        video_path = download_video(
            clean_url,
            workdir,
            cookie_file,
        )

        slides = extract_slide_images(video_path, workdir)

        output_pdf = workdir / "Lecture_Slides.pdf"

        with output_pdf.open("wb") as output:
            img2pdf.convert(slides, outputstream=output)

        # PDF ready hone par unnecessary temporary files hata dein.
        video_path.unlink(missing_ok=True)
        shutil.rmtree(workdir / "slides", ignore_errors=True)

        if cookie_file:
            Path(cookie_file).unlink(missing_ok=True)

        # Response send hone ke baad PDF folder delete hoga.
        background_tasks.add_task(
            cleanup_directory,
            str(workdir),
        )

        response = FileResponse(
            path=str(output_pdf),
            media_type="application/pdf",
            filename=f"Lecture_Slides_{video_id}.pdf",
            background=background_tasks,
        )

        response_ready = True
        return response

    except HTTPException:
        raise

    except yt_dlp.utils.DownloadError as error:
        logger.exception(
            "YouTube download failed for video %s",
            video_id,
        )

        message = str(error).lower()

        status_code = 502
        if "429" in message or "too many requests" in message:
            status_code = 503
            detail = (
                "YouTube ne server requests rate-limit ki hain (429). "
                "Abhi repeated retry na karein; kuch samay baad test karein."
            )
        elif "requested format is not available" in message:
            detail = (
                "Downloadable video format nahi mila. Server logs mein "
                "JavaScript challenge/EJS aur access warnings check karein."
            )
        elif "not a bot" in message or "sign in" in message:
            detail = (
                "YouTube ne server request par sign-in verification "
                "maangi hai. Server cookies missing, expired ya "
                "reject ho sakti hain. Valid cookies ke baad bhi "
                "YouTube access restrictions reh sakti hain."
            )
        else:
            detail = (
                "YouTube video download nahi hui. "
                "Video unavailable, restricted ya processing "
                "limit se bahar ho sakti hai. Server logs check karein."
            )

        raise HTTPException(
            status_code=status_code,
            detail=detail,
        )

    except Exception:
        logger.exception(
            "Slide processing failed for video %s",
            video_id,
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "Video processing mein error aaya. "
                "Server logs check karein."
            ),
        )

    finally:
        if not response_ready:
            cleanup_directory(str(workdir))
