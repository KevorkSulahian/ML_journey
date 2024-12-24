from pydantic import BaseModel
from crewai.flow.flow import Flow, listen, start
from crews.video_crew.video_crew import VideoCrew
from tools.custom_tool import generate_summary_from_frames_and_transcription
from moviepy import VideoFileClip
from typing import ClassVar
import cv2
import base64
import os
from pydantic import BaseModel
from openai import OpenAI
import dotenv
dotenv.load_dotenv()

def process_video(video_path, seconds_per_frame=2):
    base64Frames = []
    base_video_path, _ = os.path.splitext(video_path)
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame = 0

    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()

    audio_path = f"{base_video_path}.mp3"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, bitrate="32k")
    clip.audio.close()
    clip.close()

    print(f"Extracted audio to {audio_path}")
    return base64Frames, audio_path


class VideoState(BaseModel):
    video_path: str = "video.mp4"
    transcription: str = ""
    base64Frames: list = []
    client: ClassVar[OpenAI] = OpenAI()
    summary: str = ""
    result: str = ""


class VideoFlow(Flow[VideoState]):
    @start()
    def generate_audio_and_frames(self):
        print("Processing Video")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        video_path = os.path.join(script_dir, self.state.video_path)
        base64Frames, audio_path = process_video(video_path, seconds_per_frame=5)

        # Define the transcription file path
        transcription_file = f"{os.path.splitext(video_path)[0]}_transcription.txt"

        if os.path.exists(transcription_file):
            print("Transcription file exists. Reading from file.")
            with open(transcription_file, "r") as file:
                self.state.transcription = file.read()
        else:
            print("Uploading audio to OpenAI")
            transcription = self.state.client.audio.transcriptions.create(
                model="whisper-1",
                file=open(audio_path, "rb"),
            )
            self.state.transcription = transcription.text

            # Save transcription to file
            with open(transcription_file, "w") as file:
                file.write(self.state.transcription)

        self.state.base64Frames = base64Frames
        print("Transcription generated")
        return True

    @listen(generate_audio_and_frames)
    def generate_summary(self):
        print("Generating summary")
        # try:
        self.state.summary = generate_summary_from_frames_and_transcription(
            self.state.base64Frames, self.state.transcription, self.state.client
        )
        print("Summary generated successfully")
        # except Exception as e:
        #     print(f"Error during summary generation: {e}")
        #     # Return False to stop the flow
        #     return False
        return True

    @listen(generate_summary)
    def analyze_summary(self):
        print("Analyzing summary")
        result = (
            VideoCrew()
            .crew()
            .kickoff(inputs={"summary": self.state.summary})
        )

        print("Result generated:", result.raw)
        self.state.result = result.raw

        # Save result as a text file
        with open("result.txt", "w") as f:
            f.write(self.state.result)


def kickoff():
    video_flow = VideoFlow()
    video_flow.kickoff()


def plot():
    video_flow = VideoFlow()
    video_flow.plot()


if __name__ == "__main__":
    kickoff()
