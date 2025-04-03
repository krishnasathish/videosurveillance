import torch
import torchvision.transforms as transforms
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import json
import cv2
import os
import numpy as np
import analyze_video_detect_objects 

class VideoEmbeddingGenerator:
    def __init__(self):
        """
        Initialize multi-modal embedding generation using CLIP and sentence transformers.
        """
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_keyframes(self, video_path, num_frames=5):
        """
        Extract key frames from a video.
        
        Args:
            video_path: Path to the video file.
            num_frames: Number of frames to extract.
        
        Returns:
            List of extracted frames.
        """
        frames = []
        video = cv2.VideoCapture(video_path)
        
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = max(1, total_frames // num_frames)
        
        for i in range(0, total_frames, frame_step):
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                if len(frames) == num_frames:
                    break
        
        video.release()
        return frames

    def generate_visual_embedding(self, frames):
        """
        Generate visual embedding using CLIP.
        
        Args:
            frames: List of video frames.
        
        Returns:
            Visual embedding vector.
        """
        processed_frames = [self.transform(frame).unsqueeze(0) for frame in frames]
        frame_tensors = torch.cat(processed_frames)
        
        with torch.no_grad():
            image_embeddings = self.clip_model.get_image_features(frame_tensors)
        
        visual_embedding = F.normalize(image_embeddings.mean(dim=0), p=2, dim=0)
        
        return visual_embedding.numpy()

    def generate_semantic_embedding(self, objects, context):
        """
        Generate semantic embedding from detected objects and context.
        
        Args:
            objects: List of detected objects.
            context: Context dictionary from previous analysis.
        
        Returns:
            Semantic embedding vector.
        """
        object_str = ", ".join([obj['label'] for obj in objects])
        primary_objects = ", ".join(context.get('primary_objects', []))
        
        semantic_text = f"Video contains: {object_str}. Primary objects: {primary_objects}"
        
        semantic_embedding = self.text_encoder.encode(semantic_text)
        
        return semantic_embedding

    def generate_clip_embedding(self, video_path, previous_analysis):
        """
        Generate multi-modal embedding for a video clip.
        
        Args:
            video_path: Path to the video file.
            previous_analysis: Analysis result from object detection.
        
        Returns:
            Comprehensive embedding dictionary.
        """
        frames = self.extract_keyframes(video_path)
        
        visual_embedding = self.generate_visual_embedding(frames)
        
        semantic_embedding = self.generate_semantic_embedding(
            previous_analysis.get('detected_objects', []), 
            previous_analysis.get('context_analysis', {})
        )
        
        combined_embedding = np.concatenate([visual_embedding, semantic_embedding])
        
        embedding_result = {
            'filename': os.path.basename(video_path),
            'visual_embedding': visual_embedding.tolist(),
            'semantic_embedding': semantic_embedding.tolist(),
            'combined_embedding': combined_embedding.tolist()
        }
        
        output_path = video_path.replace('.mp4', '_embedding.json')
        with open(output_path, 'w') as f:
            json.dump(embedding_result, f, indent=2)
        
        return embedding_result

def generate_video_embedding(input_folder, analysis_results):
    """
    Generate embeddings for all video clips in a folder.
    
    Args:
        input_folder: Folder containing video clips.
        analysis_results: Previous object detection analysis results.
    
    Returns:
        List of embedding results.
    """
    embedding_generator = VideoEmbeddingGenerator()
    embedding_results = []
    
    for result in analysis_results:
        video_path = os.path.join(input_folder, result['filename'])
        
        try:
            embedding = embedding_generator.generate_clip_embedding(video_path, result)
            embedding_results.append(embedding)
        except Exception as e:
            print(f"Error generating embedding for {result['filename']}: {e}")
    
    return embedding_results


clips_folder = 'clips' # Output directory name
results = analyze_video_detect_objects.analyze_video(clips_folder)
embeddings = generate_video_embedding(clips_folder, results)