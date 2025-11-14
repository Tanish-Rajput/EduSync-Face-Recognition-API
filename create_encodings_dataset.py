import face_recognition
import json
import os
from pathlib import Path

def train_face_dataset():
    """
    Process face_dataset directory and create encodings with metadata
    """
    source_dir = Path("face_dataset")
    output_dir = Path("trained_face_dataset")
    output_dir.mkdir(exist_ok=True)
    
    # Process each person's folder
    for person_folder in source_dir.iterdir():
        if not person_folder.is_dir():
            continue
            
        person_name = person_folder.name
        print(f"\nProcessing {person_name}...")
        
        # Load metadata
        metadata_file = person_folder / "metadata.json"
        if not metadata_file.exists():
            print(f"  No metadata.json found for {person_name}, skipping...")
            continue
            
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Collect encodings from all images
        encodings = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for img_file in person_folder.iterdir():
            if img_file.suffix.lower() not in image_extensions:
                continue
                
            try:
                image = face_recognition.load_image_file(str(img_file))
                face_encodings = face_recognition.face_encodings(image)
                
                if face_encodings:
                    encodings.append(face_encodings[0].tolist())
                    print(f" Encoded {img_file.name}")
                else:
                    print(f" No face found in {img_file.name}")
            except Exception as e:
                print(f" Error processing {img_file.name}: {e}")
        
        if not encodings:
            print(f"  No valid encodings for {person_name}")
            continue
        
        # Create output directory for this person
        person_output = output_dir / person_name
        person_output.mkdir(exist_ok=True)
        
        # Save encodings
        encodings_file = person_output / "encodings.json"
        with open(encodings_file, 'w') as f:
            json.dump(encodings, f)
        
        # Copy metadata
        output_metadata = person_output / "metadata.json"
        with open(output_metadata, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Saved {len(encodings)} encodings for {person_name}")
    
    print("\n Training complete!")

if __name__ == "__main__":
    train_face_dataset()
