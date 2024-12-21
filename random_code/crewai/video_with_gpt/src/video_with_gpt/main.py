from pydantic import BaseModel
from crewai.flow.flow import Flow, listen, start
from crews.video_crew.video_crew import VideoCrew
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
        base64Frames, audio_path = process_video(video_path, seconds_per_frame=1)

        print("Uploading audio to OpenAI")
        transcription = self.state.client.audio.transcriptions.create(
            model="whisper-1",
            file=open(audio_path, "rb"),
        )
        self.state.transcription = transcription.text
        self.state.base64Frames = base64Frames

        print("Transcription generated")
        return True

    @listen(generate_audio_and_frames)
    def generate_summary(self):
        print("Generating summary")
        
        # Limit the number of frames
        selected_frames = self.state.base64Frames[:10]  # Only use the first 10 frames
        
        # Summarize transcription if too long
        max_transcription_length = 1000  # Limit to 1000 characters
        transcription = (
            self.state.transcription[:max_transcription_length]
            if len(self.state.transcription) > max_transcription_length
            else self.state.transcription
        )
        
        # Prepare the message
        frame_text = "\n".join(
            [f"![Frame](data:image/jpg;base64,{frame})" for frame in selected_frames]
        )
        user_message = f"""
        These are the frames from the video:
        {frame_text}
        The audio transcription is: {transcription}
        """
        
        # Send the request
        response = self.state.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Generate a video summary based on the frames and transcription."},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
        )
        self.state.summary = response["choices"][0]["message"]["content"]
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
