# Multimodal Video Search with LLM Validation

This project implements a comprehensive multimodal video search and retrieval system, combining advanced techniques like CLIP embeddings, FAISS similarity search, and LLM-powered result validation to provide an enriched video search experience.

## Overview

The core of this system lies in its ability to understand and correlate both the visual and semantic content of video clips with user-provided text queries. It achieves this by generating rich multimodal embeddings, indexing them for efficient retrieval, and then leveraging a Language Model (LLM) to provide contextually relevant and opinionated results.

## Key Components and Approach

1.  **Video Clip Processing and Multimodal Embedding Generation:**
    * **Input:** The system takes a full video as input. Using OpenCV's VideoCapture, it finds fps and total number of frames. Based on the desired clip duration, it extracts the clips and creates a directory of video clips (e.g., MP4 files).
    * **Object Detection (YOLOv3):** Utilizes YOLOv3 to detect objects within video frames, extracting semantic information crucial for understanding the video's content.
    * **Keyframe Extraction:** Selects representative frames from each video to capture its visual essence, reducing computational load while preserving key information.
    * **CLIP Embeddings (Visual):** Employs CLIP (Contrastive Language-Image Pre-training) to generate high-dimensional embeddings from the extracted keyframes, capturing the visual semantics.
    * **Sentence Transformers (Semantic):** Uses sentence transformers to create embeddings from the detected objects and context, encoding the textual aspects of the video's content.
    * **Combined Embeddings:** Concatenates the visual and semantic embeddings to create a comprehensive multimodal representation of each video clip.
    * **JSON Storage:** Stores the generated embeddings and associated metadata (filenames) in JSON files for persistence and easy access.

2.  **Efficient Video Retrieval with FAISS:**
    * **FAISS Indexing:** Utilizes FAISS (Facebook AI Similarity Search) to create an efficient index of the generated video embeddings, enabling fast similarity searches.
    * **Index Creation:** Constructs a flat L2 distance index, well-suited for high-dimensional embeddings, to facilitate nearest neighbor searches.
    * **Embedding Indexing:** Adds the visual embeddings to the FAISS index, creating a searchable database of video content.
    * **Persistence:** Saves the FAISS index to disk, ensuring that the index can be reloaded for subsequent searches without re-computation.

3.  **Text-Based Video Search:**
    * **User Query Input:** Accepts text queries from users, describing the desired video content.
    * **CLIP Text Embedding:** Converts the user's text query into a CLIP embedding, aligning the query with the video embeddings in the same semantic space.
    * **FAISS Similarity Search:** Queries the FAISS index using the generated text embedding, retrieving the most similar video embeddings.
    * **Result Retrieval:** Retrieves the filenames of the top-k matching video clips based on the FAISS search results.

4.  **LLM-Powered Result Validation and Opinion Generation:**
    * **LLM Integration (Ollama with Mistral):** Integrates with gemini-2.0-flash to provide contextual analysis and opinions on the search results.
    * **Prompt Engineering:** Employs a carefully designed prompt to instruct the LLM to extract and present the insights in a structured format:
        * Summary of Events
        * Behavioral Patterns
        * Alerts & Notifications**
        * Identifications & Profiles
        * Potential Threat Assessment
    * **LLM Processing:** Passes the retrieved video clip and the user's query to the LLM for processing.
    * **Response Generation:** Generates a comprehensive response that includes the LLM's analysis, opinions, and summaries.

5.  **Streamlit User Interface:**
    * **User-Friendly Interface:** Provides a simple and intuitive Streamlit-based UI for text-based video searches.
    * **Query Input:** Allows users to enter their search queries via a text input field.
    * **Video Display:** Displays the retrieved video clips within the UI.
    * **LLM Output Display:** Presents the LLM's response, including analysis and opinions, to the user.
    * **Error Handling:** Implements robust error handling for missing video files and cases where no relevant videos are found.

## Usage and Implementation Details

* **Dependencies:** Requires Python 3.6+, CUDA (optional), FFmpeg, and Ollama with a suitable model (e.g., Mistral).
* **Installation:** Involves cloning the repository, installing dependencies, downloading models, preparing video clips, generating embeddings, and running the Streamlit application.
* **Workflow:** Users input text queries, the system retrieves relevant videos using FAISS, and the LLM provides context and opinions, all displayed through a Streamlit interface.

This approach provides a robust and intelligent video search system, combining the power of multimodal embeddings and LLM reasoning to deliver highly relevant and informative search results.
