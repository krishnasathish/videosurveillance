import cv2
import numpy as np
import os
import json
from pathlib import Path
import urllib.request  

class VideoClipAnalyzer:
    def __init__(self):
        """
        Initialize video clip analysis with modern object detection model (YOLOv3).
        """
        # Pre-trained model paths
        model_dir = Path.home() / '.opencv_models'
        model_dir.mkdir(exist_ok=True)
        
        # Download pre-trained model files if they don't exist
        self._download_model_files(model_dir)
        
        # Load YOLO model
        self.net = cv2.dnn.readNetFromDarknet(
            str(model_dir / 'yolov3.cfg'),
            str(model_dir / 'yolov3.weights')
        )
        
        # Set preferred backend and target for inference
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Load COCO class names
        classes_file = model_dir / 'coco.names'
        with open(str(classes_file), 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

    def _download_model_files(self, model_dir):
        """
        Download necessary YOLO model files if they don't exist.
        """
        model_files = {
            'yolov3.cfg': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
            'yolov3.weights': 'https://pjreddie.com/media/files/yolov3.weights',
            'coco.names': 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
        }
        
        for filename, url in model_files.items():
            file_path = model_dir / filename
            if not file_path.exists():
                print(f"Downloading {filename}...")
                try:
                    urllib.request.urlretrieve(url, file_path)
                    print(f"Downloaded {filename}")
                except Exception as e:
                    print(f"Error downloading {filename}: {e}")
                    raise

    def _get_output_layers(self):
        """
        Get the output layer names of the YOLO network.
        """
        layer_names = self.net.getLayerNames()
        
        # Different OpenCV versions have different methods to get output layers
        try:
            return [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        except:
            return [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_objects(self, frame):
        """
        Detect objects in a single frame using YOLO.
        
        Args:
            frame: Input image frame.
        
        Returns:
            List of detected objects with confidence scores and bounding boxes.
        """
        height, width = frame.shape[:2]
        
        # Prepare frame for YOLO detection (416x416 is a standard input size)
        blob = cv2.dnn.blobFromImage(
            frame, 1/255.0, (416, 416), 
            swapRB=True, 
            crop=False
        )
        
        # Run forward pass to get detections
        self.net.setInput(blob)
        outputs = self.net.forward(self._get_output_layers())
        
        # Initialize lists to store detection results
        boxes = []
        confidences = []
        class_ids = []
        
        # Process each output layer
        for output in outputs:
            for detection in output:
                # Extract class scores from detection (first 5 elements are box info)
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter weak detections
                if confidence > 0.5:
                    # Scale bounding box coordinates to image size
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression to remove redundant detections
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        objects = []
        if len(indices) > 0:
            indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
            
            for i in indices:
                box = boxes[i]
                objects.append({
                    'label': self.classes[class_ids[i]],
                    'confidence': confidences[i],
                    'bbox': box
                })
        
        return objects

    def analyze_clip(self, clip_path):
        """
        Analyze a video clip for objects and context, saving the results to a JSON file.
        
        Args:
            clip_path: Path to the video clip.
        
        Returns:
            Comprehensive analysis of the video clip.
        """
        video = cv2.VideoCapture(clip_path)
        
        # Robust FPS and duration calculation
        fps = video.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # Fallback to 30 FPS if unable to retrieve
        
        frame_count = 0
        all_objects = []
        object_counts = {}
        
        max_objects_to_store = 100  # Limit object storage to prevent oversized JSON
        
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            # Process every 5th frame to reduce processing load
            if frame_count % 5 == 0:
                frame_objects = self.detect_objects(frame)
                
                if len(all_objects) < max_objects_to_store:
                    all_objects.extend(frame_objects)
                
                for obj in frame_objects:
                    label = obj['label']
                    object_counts[label] = object_counts.get(label, 0) + 1
            
            frame_count += 1
        
        video.release()
        
        duration = frame_count / fps
        
        top_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        analysis_result = {
            'filename': os.path.basename(clip_path),
            'total_frames': frame_count,
            'fps': fps,
            'duration': duration,
            'context_analysis': {
                'primary_objects': [obj[0] for obj in top_objects],
                'object_diversity': len(object_counts),
                'total_objects_detected': sum(object_counts.values()),
                'average_objects_per_frame': sum(object_counts.values()) / frame_count if frame_count > 0 else 0
            },
            'detected_objects': all_objects
        }
        
        output_json = clip_path.replace('.mp4', '_analysis.json')
        
        try:
            with open(output_json, 'w') as f:
                json.dump(analysis_result, f, indent=2)
            
            if os.path.exists(output_json):
                with open(output_json, 'r') as f:
                    json.load(f)
        except Exception as e:
            print(f"Error writing JSON for {clip_path}: {e}")
            
            simplified_result = {
                'filename': os.path.basename(clip_path),
                'total_frames': frame_count,
                'fps': fps,
                'duration': duration,
                'context_analysis': analysis_result['context_analysis'],
                'detected_objects': all_objects[:20]
            }
            
            try:
                with open(output_json, 'w') as f:
                    json.dump(simplified_result, f, indent=2)
            except Exception as e:
                print(f"Failed to write even simplified JSON: {e}")
        
        return analysis_result

def repair_json_files(directory):
    """
    Check and repair any broken JSON files in the directory.
    
    Args:
        directory: Directory containing JSON files.
    """
    for filename in os.listdir(directory):
        if filename.endswith('_analysis.json'):
            file_path = os.path.join(directory, filename)
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                continue
            except json.JSONDecodeError:
                print(f"Found corrupted JSON file: {file_path}")
                
                with open(file_path, 'r') as f:
                    partial_data = f.read()
                
                last_complete = partial_data.rfind("},")
                
                if last_complete > 0:
                    repaired_data = partial_data[:last_complete+1] + "]}"
                    
                    brackets_to_close = repaired_data.count("{") - repaired_data.count("}")
                    if brackets_to_close > 0:
                        repaired_data += "}" * brackets_to_close
                    
                    try:
                        json.loads(repaired_data)
                        
                        with open(file_path + '.repaired', 'w') as f:
                            f.write(repaired_data)
                        
                        print(f"Successfully repaired JSON to {file_path}.repaired")
                    except json.JSONDecodeError:
                        print(f"Could not repair JSON file: {file_path}")
                else:
                    print(f"Could not find a valid repair point in {file_path}")

def process_surveillance_clips(input_folder):
    """
    Process all video clips in a folder and return analysis results.
    
    Args:
        input_folder: Folder containing video clips.
    
    Returns:
        List of analysis results.
    """
    analyzer = VideoClipAnalyzer()
    analysis_results = []
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.mp4'):
            clip_path = os.path.join(input_folder, filename)
            try:
                result = analyzer.analyze_clip(clip_path)
                analysis_results.append(result)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    repair_json_files(input_folder)
    
    return analysis_results

input_clips_folder = 'video_clips'
results = process_surveillance_clips(input_clips_folder)

for result in results:
    print(f"Clip: {result['filename']}")
    print(f"Duration: {result['duration']:.2f} seconds")
    print(f"Total Frames: {result['total_frames']}")
    print(f"FPS: {result['fps']}")
    print(f"Primary Objects: {result['context_analysis']['primary_objects']}")
    print(f"Total Objects Detected: {result['context_analysis']['total_objects_detected']}")
    print("---")

import torch
import torchvision.transforms as transforms
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

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

def generate_video_embeddings(input_folder, analysis_results):
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

input_clips_folder = 'video_clips'
previous_analysis_results = process_surveillance_clips(input_clips_folder)
embeddings = generate_video_embeddings(input_clips_folder, previous_analysis_results)

for embedding in embeddings:
    print(f"Clip: {embedding['filename']}")
    print(f"Visual Embedding Length: {len(embedding['visual_embedding'])}")
    print(f"Semantic Embedding Length: {len(embedding['semantic_embedding'])}")
    print(f"Combined Embedding Length: {len(embedding['combined_embedding'])}")
    print("---")
