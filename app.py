import streamlit as st
import utils.faiss_db_utils as faiss
import utils.llm_utils as llm
import os



# Streamlit UI: Create a user interface for video search.
st.title("🎥 Video Search with Multimodal RAG")
query = st.text_input("Enter your search query:")

if st.button("Search"):
    with st.spinner("Retrieving videos..."):
        retrieved_clips = faiss.search_videos_by_text(query)  # Retrieve videos based on query.
        llm_response = llm.validate_with_llm(query, retrieved_clips)  # Validate and rank with LLM (with opinion).

        if retrieved_clips:
            st.subheader("🔍 Retrieved Video Clips")
            for clip in retrieved_clips:
                video_path = f"clips/{clip}"
                if os.path.exists(video_path):
                    st.write(f"**{clip}**")
                    st.video(video_path)  # Display the video.
                else:
                    st.warning(f"❌ Video file missing: {video_path}")  # Handle missing files.
        else:
            st.warning("No relevant videos found.")  # Inform user if no videos are found.

        st.subheader("🧠 LLM Validation, Ranking, and Opinion")
        st.write(llm_response)  # Display LLM's response.