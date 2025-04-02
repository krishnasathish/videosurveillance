import os
import cv2
import subprocess
import shutil

def extract_and_convert_clips(input_video_path, output_folder, clip_duration=10):
    """
    Extracts clips of a specified duration from an input video, converts them to H.264 using FFmpeg, 
    and saves them to an output folder.

    Args:
        input_video_path (str): The path to the input video file.
        output_folder (str): The path to the output folder where clips will be saved.
        clip_duration (int, optional): The duration of each clip in seconds. Defaults to 10.

    Returns:
        list: A list of paths to the extracted and converted clip files.
    """

    # Ensure the output folder exists
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)  # Remove all existing files in the folder to start clean
    os.makedirs(output_folder, exist_ok=True)  # Recreate the output folder (if it doesn't exist)

    # Open the input video file using OpenCV's VideoCapture
    video = cv2.VideoCapture(input_video_path)
    
    # Get video properties: frames per second (fps) and total number of frames
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the number of frames for each clip based on the desired duration
    frames_per_clip = int(clip_duration * fps)
    
    # Initialize an empty list to store the paths of the output clip files
    output_clips = []
    
    # Initialize variables to keep track of the current frame and clip count
    current_frame = 0
    clip_count = 0
    
    # Loop through the video frames until all frames have been processed
    while current_frame < total_frames:
        # Create a temporary filename for the raw clip (before FFmpeg conversion)
        temp_clip_filename = os.path.join(output_folder, f'temp_clip_{clip_count}.mp4')
        
        # Create the final filename for the converted clip (after FFmpeg conversion)
        fixed_clip_filename = os.path.join(output_folder, f'clip_{clip_count}.mp4')

        # Get the video codec and create a VideoWriter object to write the clip frames
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use mp4v codec for intermediate file.
        out = cv2.VideoWriter(temp_clip_filename, fourcc, fps, 
                               (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                                int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        
        # Write frames to the temporary clip file
        for _ in range(frames_per_clip):
            ret, frame = video.read() # Read a frame from the video
            if not ret: # If no frame is read (end of video), break the loop
                break
            out.write(frame) # Write the frame to the temporary clip file
            current_frame += 1 # Increment the current frame counter
        
        # Release the VideoWriter object for the current clip
        out.release()
        
        # Convert the raw clip to H.264 using FFmpeg (to ensure compatibility)
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", temp_clip_filename, # Input file
            "-vcodec", "libx264", "-acodec", "aac", "-strict", "experimental", # Video and audio codecs
            "-preset", "fast", "-b:v", "1000k", "-b:a", "128k", fixed_clip_filename # Output file, bitrate, and preset
        ]
        # Run the FFmpeg command, suppressing output to console
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Remove the temporary raw clip file
        os.remove(temp_clip_filename)
        
        # Append the path of the converted clip to the list of output clips
        output_clips.append(fixed_clip_filename)
        
        # Increment the clip counter
        clip_count += 1

    # Release the VideoCapture object for the input video
    video.release()
    
    # Return the list of output clip file paths
    return output_clips

# Example usage
input_video = 'surveillance.mp4' # Input video filename
output_dir = 'video_clips' # Output directory name
clips = extract_and_convert_clips(input_video, output_dir) # Call the function to extract and convert clips

# Print the number of extracted clips and the output directory
print(f"Extracted {len(clips)} video clips in {output_dir}")
