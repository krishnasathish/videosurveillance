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

option = st.selectbox("Choose output type:", ["Text", "Image"])

prompt = st.text_area("Enter a prompt:")

if st.button("Generate Image"):
    if not prompt:
        st.warning("Please enter the prompt!")
    else:
        try:
            with st.spinner("Generating image..."):
                response = client.models.generate_content(
                model="gemini-2.0-flash-exp-image-generation",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=['Text', 'Image']
                )
            )
            
            st.subheader("Generated Image")
            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    st.write(part.text)
                elif part.inline_data is not None:
                    image = Image.open(BytesIO((part.inline_data.data)))
                    st.image(image)
                
        except Exception as e:
            st.error(f"Error generating image {e}")
