import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
import io
from io import BytesIO

# Load API key
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Streamlit UI
st.title("Gemini AI: Text & Image Generator")

st.title("AI Theft YouTube Video Summarizer")
youtube_url = st.text_input("Enter YouTube Video URL")

if st.button("Summarize Video"):
    if not youtube_url:
        st.warning("No YouTube URL Present!")
    else:
        try:
            with st.spinner("Anaylizing video..."):
                response = client.models.generate_content(
                    model='models/gemini-2.0-flash',
                    contents=types.Content(
                        parts=[
                            types.Part(text='Can you highlight any suspicious behavior and provide the timestamp?'),
                            types.Part(
                                file_data=types.FileData(file_uri=youtube_url)
                            )
                        ]
                    )
                )
            st.subheader("Video analysis Summary")
            st.write(response.text)
        except Exception as e:
            st.error("Error generating summary")
