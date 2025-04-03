import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
import io
from moviepy import VideoFileClip
import yt_dlp
import cv2
import numpy as np
from io import BytesIO
import re

# Load API key
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

st.title("AI Theft Detection Tool")

# Input for YouTube URL
youtube_url = st.text_input("Enter YouTube Video URL")

if st.button("Summarize Video"):
    if not youtube_url:
        st.warning("No YouTube URL Present!")
    else:
        try:
            with st.spinner("Downloading transcript and analyzing..."):
                # Step 1: Extract transcript using Whisper or YouTube API
                transcript_text = "Extracted transcript goes here..."

                # Step 2: Ask Gemini to analyze for suspicious behavior & timestamps
                response = client.models.generate_content(
                    model='models/gemini-2.0-flash',
                    contents=types.Content(
                        parts=[
                            types.Part(text=f"Analyze this transcript for suspicious behavior and provide timestamps: {transcript_text}"),
                            types.Part(
                                file_data=types.FileData(file_uri=youtube_url)
                            )
                        ]
                    )
                )

                st.subheader("Video Analysis Summary")
                st.write(response.text)

                # Step 3: Extract timestamps from response
                timestamps = [line for line in response.text.split("\n") if ":" in line]  # Extract timestamps

        except Exception as e:
            st.error(f"Error analyzing video: {e}")

        # Step 4: Capture Screenshots at Timestamps
        try:
            with st.spinner("Downloading video and capturing frames..."):
                # Download Video using yt-dlp
                ydl_opts = {
                    'format': 'best',
                    'outtmpl': 'video.mp4'
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([youtube_url])

                # Load video and capture frames
                clip = VideoFileClip("video.mp4")

                st.subheader("Screenshots from Suspicious Timestamps")
                for ts in timestamps:
                    try:
                        print(f"Timestamp: {ts}")
                        timestamp_pattern = r"(\d{1,2}:\d{2})"
                        matches = re.findall(timestamp_pattern, ts)            

                        # 🔹 Step 2: Convert timestamps to seconds
                        timestamps_arr = []
                        for match in matches:
                            minutes, seconds = map(int, match.split(":"))
                            timestamps_arr.append(minutes * 60 + seconds)

                        print("Extracted Timestamps (in seconds):", timestamps_arr)

                        # Extract Frame
                        for timestamp_sec in timestamps_arr:
                            if timestamp_sec < clip.duration:
                                frame = clip.get_frame(timestamp_sec)
                                image = Image.fromarray(frame)
                                st.image(image, caption=f"Screenshot at {timestamp_sec} seconds")

                    except Exception as e:
                        st.warning(f"Failed to capture screenshot at {ts}: {e}")

        except Exception as e:
            st.error(f"Error processing video frames: {e}")
