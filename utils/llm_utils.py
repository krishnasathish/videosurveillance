import google.generativeai as genai
import base64

def validate_with_llm(query, retrieved_clips):
    """
    Pass retrieved video clips to the LLM for ranking and validation, including opinion.

    Args:
        query (str): The original search query.
        retrieved_clips (list): List of retrieved video filenames.

    Returns:
        str: The LLM's response, including opinions and analysis.
    """
    genai.configure(api_key="")
    llm = genai.GenerativeModel('models/gemini-2.0-flash')
    video_file_name = f"clips/{retrieved_clips[0]}"
    video_bytes = open(video_file_name, 'rb').read()

    # Encode video bytes to base64
    video_base64 = base64.b64encode(video_bytes).decode('utf-8')

    # Prompt for LLM Validation: Define a prompt for ranking and validating retrieved videos.
    prompt = f"""
    You are an AI security analyst. Given the user's query: "{query}", analyze the retrieved surveillance data below.
    
    Retrieved Surveillance Data:
    {retrieved_clips}
    Transcribe the audio from this video, giving timestamps for salient events in the video. Also provide visual descriptions.
    
    Extract and present the insights in a structured format:
    - **Summary of Events**
    - **Behavioral Patterns**
    - **Alerts & Notifications**
    - **Identifications & Profiles**
    - **Potential Threat Assessment**
    
    Ensure the response is clear, concise, and actionable.
    """
    # Create the content payload
    contents = [
        {
            "mime_type": "video/mp4",  # Important: Specify the MIME type
            "data": video_base64,
        },
        prompt,
    ]
    response = llm.generate_content(contents)
    return response.text
