import face_recognition
from fastapi import FastAPI, HTTPException
import json
from pathlib import Path
import numpy as np
from supabase import create_client, Client
from pydantic import BaseModel
import os
from typing import List, Dict
import tempfile
import uvicorn

supabase_url = 'https://lahrdlkzclbcnordtckv.supabase.co'
supabase_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxhaHJkbGt6Y2xiY25vcmR0Y2t2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDA2NDE4MDgsImV4cCI6MjA1NjIxNzgwOH0.Vj3SsuLeCaJvfmrNEw4BcLan63LaKWmtjKGWvU33eAc'
supabase: Client = create_client(supabase_url, supabase_key)

def recognize_face(image_path, tolerance=0.6):
    trained_dir = Path("trained_face_dataset")
    
    # Load and encode the unknown image
    print(f"Loading image: {image_path}")
    try:
        unknown_image = face_recognition.load_image_file(image_path)
        unknown_encodings = face_recognition.face_encodings(unknown_image)
        
        if not unknown_encodings:
            print("No face detected in the image!")
            return None
        
        unknown_encoding = unknown_encodings[0]
        print(f"Face detected and encoded\n")
    
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    # Search through all trained persons
    best_match = None
    best_distance = float('inf')
    
    print("Comparing against trained faces...\n")
    
    for person_folder in trained_dir.iterdir():
        if not person_folder.is_dir():
            continue
        
        person_name = person_folder.name
        
        # Load encodings
        encodings_file = person_folder / "encodings.json"
        metadata_file = person_folder / "metadata.json"
        
        if not encodings_file.exists() or not metadata_file.exists():
            continue
        
        with open(encodings_file, 'r') as f:
            stored_encodings = json.load(f)
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Convert back to numpy arrays
        stored_encodings = [np.array(enc) for enc in stored_encodings]
        
        # Compare with all stored encodings for this person
        distances = face_recognition.face_distance(stored_encodings, unknown_encoding)
        min_distance = float(np.min(distances))
        
        print(f"  {person_name}: distance = {min_distance:.4f}")
        
        # Check if this is the best match
        if min_distance < best_distance and min_distance <= tolerance:
            best_distance = min_distance
            best_match = {
                'metadata': metadata,
                'distance': min_distance,
                'person_name': person_name
            }
    
    print()
    
    if best_match:
        print("MATCH FOUND!")
        print(f"Person: {best_match['person_name']}")
        print(f"Distance: {best_match['distance']:.4f}")
        print(f"Metadata: {json.dumps(best_match['metadata'], indent=2)}")
        return best_match
    else:
        print("No match found (or distance exceeds tolerance)")
        return None

def recognize_faces_batch(image_paths, tolerance=0.6):
    """
    Recognize multiple faces from a list of images
    """
    results = []
    for img_path in image_paths:
        print("="*60)
        result = recognize_face(img_path, tolerance)
        results.append({'image': img_path, 'match': result})
        print()
    
    return results

app = FastAPI(title="Face Recognition API")

class RecognitionRequest(BaseModel):
    subjectName: str
    timestamp: str

def process_images_for_recognition(image_paths: List[str]) -> Dict:
    all_recognized_people = {}
    image_results = []
    
    for image_path in image_paths:
        try:
            # recognize_face returns a single dict or None
            result = recognize_face(image_path)
            
            if result:
                image_result = {
                    "image_path": os.path.basename(image_path),
                    "faces_found": 1,
                    "person_name": result["person_name"],
                    "distance": result["distance"],
                    "metadata": result["metadata"]
                }
                
                # Aggregate (using distance, lower is better)
                person_name = result["person_name"]
                if person_name not in all_recognized_people or result["distance"] < all_recognized_people[person_name]["distance"]:
                    all_recognized_people[person_name] = {
                        "distance": result["distance"],
                        "first_seen_in": os.path.basename(image_path),
                        "metadata": result["metadata"]
                    }
            else:
                image_result = {
                    "image_path": os.path.basename(image_path),
                    "faces_found": 0
                }
            
            image_results.append(image_result)
                    
        except Exception as e:
            image_results.append({
                "image_path": os.path.basename(image_path),
                "error": str(e)
            })
    
    return {
        "image_results": image_results,
        "all_recognized_people": all_recognized_people
    }

def build_folder_query(subject_name: str, timestamp: str) -> str:
    return f"{subject_name}_{timestamp}"

def fetch_images_from_supabase(folder_name: str) -> List[str]:
    try:
        # List all files in the folder from Supabase storage
        # Adjust 'your-bucket-name' to your actual bucket name
        bucket_name = "your-bucket-name"
        
        # List files in the folder
        files = supabase.storage.from_(bucket_name).list(folder_name)
        
        if not files:
            return []
        
        # Download images to temp directory
        temp_dir = tempfile.mkdtemp()
        local_image_paths = []
        
        for file in files:
            # Skip if not an image file
            if not file['name'].lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                continue
                
            # Build the full file path in storage
            file_path = f"{folder_name}/{file['name']}"
            
            # Download the file
            response = supabase.storage.from_(bucket_name).download(file_path)
            
            # Save to temp file
            local_path = os.path.join(temp_dir, file['name'])
            with open(local_path, 'wb') as f:
                f.write(response)
            
            local_image_paths.append(local_path)
        
        return local_image_paths
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching images: {str(e)}")

@app.post("/recognize")
async def recognize_faces(request: RecognitionRequest):
    try:
        # Build the folder query
        folder_name = build_folder_query(request.subjectName, request.timestamp)
        
        # Fetch images from Supabase
        images = fetch_images_from_supabase(folder_name)
        
        if not images:
            return {
                "status": "error",
                "message": f"No images found in folder: {folder_name}"
            }
        
        recognition_results = process_images_for_recognition(images)
        
        # Cleanup temp files
        # for path in image_paths:
        #     if os.path.exists(path):
        #         os.remove(path)
        
        return {
            "status": "success",
            "folder_name": folder_name,
            "total_images_processed": len(images),
            "total_people_recognized": len(recognition_results["all_recognized_people"]),
            "results": recognition_results
        }
        
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Optional: Add a cleanup endpoint
@app.post("/cleanup")
async def cleanup_temp_files():
    """Clean up temporary downloaded files"""
    try:
        temp_dir = tempfile.gettempdir()
        # Add your cleanup logic here
        return {"status": "success", "message": "Temp files cleaned"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup error: {str(e)}")
    
@app.get("/")
def greet():
    return {"message": "Hello World"}
    

if __name__ == "__main__":
    uvicorn.run("recog_face:app", host="127.0.0.1", port=8000, reload=True)