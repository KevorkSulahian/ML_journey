#!/usr/bin/env python
from random import randint

from pydantic import BaseModel

from crewai.flow.flow import Flow, listen, start

from crews.video_crew.video_crew import VideoCrew

import cv2
from moviepy import VideoFileClip
import time
import base64
import os
from typing import ClassVar
from pydantic import BaseModel
from openai import OpenAI, ChatCompletion
import dotenv
dotenv.load_dotenv()


def process_video(video_path, seconds_per_frame=2):
    base64Frames = []
    base_video_path, _ = os.path.splitext(video_path)
    print(f"Processing video at {video_path}")
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



class VideoState(BaseModel):
    video_path: str = "video.mp4"
    transcription: str = ""
    base64Frames: list = []
    client: ClassVar[OpenAI] = OpenAI()
    summary: str = ""


class VideoFlow(Flow[VideoState]):
    
    @start()
    def generate_audio_and_frames(self):
        print("Processing Video")
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
        video_path = os.path.join(script_dir, self.state.video_path) 
        base64Frames, audio_path = process_video(video_path, seconds_per_frame=1)

        print("Uploading audio to OpenAI")
        transcription = self.state.client.audio.transcriptions.create(
            model="whisper-1",
            file=open(audio_path, "rb"),
        )
        self.state.transcription = transcription.text
        self.state.base64Frames = base64Frames

        print("Transcription generated")

    @listen(generate_audio_and_frames)
    def generate_summary(self):
        print("Generating summary")
        response = self.state.client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
            {"role": "system", "content":"""You are generating a video summary. Create a summary of the provided video and its transcript. Respond in Markdown"""},
            {"role": "user", "content": [
                "These are the frames from the video.",
                *map(lambda x: {"type": "image_url", 
                                "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, self.state.base64Frames),
                {"type": "text", "text": f"The audio transcription is: {self.state.transcription}"}
                ],
            }
        ],
            temperature=0,
        )
        self.state.summary = response.choices[0].message.content

        

    @listen(generate_summary)
    def generate_summary(self):
        print("analyzing summary")
        result = (
            VideoCrew()
            .crew()
            .kickoff(inputs={"summary": self.state.summary})
        )

        print("result generated", result.raw)
        self.state.result = result.raw

        # save as txt
        with open("result.txt", "w") as f:
            f.write(self.state.result)


def kickoff():
    poem_flow = VideoFlow()
    poem_flow.kickoff()


def plot():
    poem_flow = VideoFlow()
    poem_flow.plot()


if __name__ == "__main__":
    kickoff()
