import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
import json
import numpy as np
import os
import streamlit as st
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# FAISS Configuration: Ensure single-threading to prevent potential crashes.
faiss.omp_set_num_threads(1)

# Load Video Filenames: Read the list of video filenames from a JSON file.
with open("video_filenames.json", "r") as f:
    filenames = json.load(f)

# Load Video Embeddings: Retrieve pre-computed video embeddings from JSON files.
folder_path = "video_clips/"
embeddings_data = []

for filename in os.listdir(folder_path):
    if filename.endswith("_embedding.json"):  # Process only embedding JSON files.
        with open(os.path.join(folder_path, filename), "r") as f:
            video_data = json.load(f)
            embeddings_data.append(video_data)

print(f"Loaded {len(embeddings_data)} video embeddings.")

# Prepare Data for FAISS: Extract visual embeddings and create a FAISS index.
embeddings = np.array([d["visual_embedding"] for d in embeddings_data], dtype=np.float32)
d = embeddings.shape[1]  # Determine the embedding dimension.
index = faiss.IndexFlatL2(d)  # Initialize a flat L2 distance index.
index.add(embeddings)  # Add embeddings to the FAISS index.

# Save FAISS Index: Persist the FAISS index to disk for future use.
faiss.write_index(index, "video_embeddings.index")
print("FAISS index saved.")

# Load CLIP Model: Initialize the CLIP model and processor for text embedding generation.
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP model loaded.")

def get_text_embedding(text):
    """
    Convert a text query into a CLIP embedding.

    Args:
        text (str): The text query to encode.

    Returns:
        numpy.ndarray: The text embedding as a NumPy array.
    """
    print("Text query:", text)
    inputs = processor(text=text, return_tensors="pt", padding=True)  # Preprocess the text.

    # Move model and inputs to GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate text embedding using CLIP.
    text_embedding = clip_model.get_text_features(**inputs)

    return text_embedding.detach().cpu().numpy().astype("float32")  # Return as NumPy array.

def search_videos_by_text(query_text, top_k=3):
    """
    Search for videos in the FAISS index using a text query.

    Args:
        query_text (str): The text query for video retrieval.
        top_k (int): The number of top results to retrieve.

    Returns:
        list: A list of filenames corresponding to the top matching videos.
    """
    query_embedding = get_text_embedding(query_text)  # Generate text embedding.
    print("Query embedding shape:", query_embedding.shape)
    print("FAISS index dimension:", index.d)
    distances, indices = index.search(query_embedding, top_k)  # Perform FAISS search.

    return [embeddings_data[i]["filename"] for i in indices[0] if i < len(embeddings_data)]  # Return filenames.

# Initialize Language Model (LLM): Set up Ollama with the Mistral model.
llm = Ollama(model="mistral")

# Prompt for LLM Validation: Define a prompt for ranking and validating retrieved videos.
prompt = PromptTemplate.from_template("""
You are an expert video analyst. A user searched for: "{query}". 
These video clips were retrieved: {retrieved_clips}.

Based on the query and the content of the retrieved video clips, provide your opinion on:

1.  **Relevance Ranking:** Rank the retrieved clips from most to least relevant to the user's query. Explain your reasoning for each ranking, focusing on specific details from the video content that support your assessment.
2.  **Irrelevance Explanation:** If any clips are irrelevant, clearly explain why they do not match the query, highlighting the discrepancies between the query and the video content.
3.  **Content Summary:** Provide a short summary of the key content or events depicted in each relevant video clip.
4.  **Overall Opinion:** Offer your overall opinion on the quality of the search results and suggest any potential improvements to the retrieval process.

Your response should be detailed, analytical, and provide a clear justification for your opinions.
""")

def validate_with_llm(query, retrieved_clips):
    """
    Pass retrieved video clips to the LLM for ranking and validation, including opinion.

    Args:
        query (str): The original search query.
        retrieved_clips (list): List of retrieved video filenames.

    Returns:
        str: The LLM's response, including opinions and analysis.
    """
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(query=query, retrieved_clips=retrieved_clips)

# Streamlit UI: Create a user interface for video search.
st.title("🎥 Video Search with Multimodal RAG")
query = st.text_input("Enter your search query:")

if st.button("Search"):
    with st.spinner("Retrieving videos..."):
        retrieved_clips = search_videos_by_text(query)  # Retrieve videos based on query.
        llm_response = validate_with_llm(query, retrieved_clips)  # Validate and rank with LLM (with opinion).

        if retrieved_clips:
            st.subheader("🔍 Retrieved Video Clips")
            for clip in retrieved_clips:
                video_path = f"video_clips/{clip}"
                if os.path.exists(video_path):
                    st.write(f"**{clip}**")
                    st.video(video_path)  # Display the video.
                else:
                    st.warning(f"❌ Video file missing: {video_path}")  # Handle missing files.
        else:
            st.warning("No relevant videos found.")  # Inform user if no videos are found.

        st.subheader("🧠 LLM Validation, Ranking, and Opinion")
        st.write(llm_response)  # Display LLM's response.
