import face_recognition
from pathlib import Path
import numpy as np
import json


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


if __name__ == "__main__":
    image = "test_image.jpg"
    recognize_face(image)
