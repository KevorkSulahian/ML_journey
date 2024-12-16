#!/usr/bin/env python
import sys
import warnings

from video_analysis.crew import VideoAnalysis

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

import cv2
from moviepy import VideoFileClip
import time
import base64
import os

VIDEO_PATH = "video.mp4"

from openai import OpenAI, ChatCompletion
import dotenv
dotenv.load_dotenv()
client = OpenAI()


def process_video(video_path, seconds_per_frame=2):
    base64Frames = []
    base_video_path, _ = os.path.splitext(video_path)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame=0

    # Loop through the video and extract frames at specified sampling rate
    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()

    # Extract audio from video
    audio_path = f"{base_video_path}.mp3"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, bitrate="32k")
    clip.audio.close()
    clip.close()

    print(f"Extracted {len(base64Frames)} frames")
    print(f"Extracted audio to {audio_path}")
    return base64Frames, audio_path

# Extract 1 frame per second. You can adjust the `seconds_per_frame` parameter to change the sampling rate



def run():
    """
    Run the crew.
    """
    print("OS path: ", os.path.abspath(os.getcwd()))
    base64Frames, audio_path = process_video(VIDEO_PATH, seconds_per_frame=1)

    transcription = client.audio.transcriptions.create(
    model="whisper-1",
    file=open(audio_path, "rb"),
)
    inputs = {
        'video_frames': base64Frames,
        'audio': transcription
    }
    print("Running VideoAnalysis")
    VideoAnalysis().crew().kickoff(inputs=inputs)
