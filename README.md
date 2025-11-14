# Face Recognition API Module

> **Part of EduSync Project** - An automatic attendance marking system using advanced face recognition technology.

**⚠️ Important Notice:** This repository contains only the Face Recognition API module of the larger EduSync project. The complete EduSync system includes additional features and functionalities that are currently under patent protection and cannot be disclosed publicly at this time.

## Overview

This Face Recognition API is a FastAPI-based RESTful service that provides real-time face recognition capabilities. It uses the `face_recognition` library (built on dlib's state-of-the-art face recognition) to identify individuals from images stored in Supabase cloud storage.

### Key Features

- **High Accuracy Recognition**: Uses deep learning-based face encodings with configurable tolerance
- **Batch Processing**: Process multiple images in a single request
- **Cloud Integration**: Seamless integration with Supabase storage
- **RESTful API**: Easy-to-use HTTP endpoints for face recognition
- **Metadata Support**: Store and retrieve additional information with each person's profile
- **Pre-trained Dataset**: Includes sample dataset for immediate testing

## Project Structure

```
face-recognition-api/
│
├── face_dataset/                    # Source images for training
│   ├── person_1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── metadata.json           # Person's information
│   └── person_2/
│       ├── image1.jpg
│       └── metadata.json
│
├── trained_face_dataset/            # Encoded face data (generated)
│   ├── person_1/
│   │   ├── encodings.json          # Face encodings (128-d vectors)
│   │   └── metadata.json           # Person's metadata
│   └── person_2/
│       ├── encodings.json
│       └── metadata.json
│
├── create_encoding_dataset.py       # Training script
├── face_recog_test_local.py        # Local testing utility
├── face_recog_API.py               # Main FastAPI application
├── test_image.jpg                  # Sample test image
└── README.md
```

## How It Works

### 1. **Training Phase** (`create_encoding_dataset.py`)

The training phase processes face images and creates numerical representations (encodings) for recognition:

```
face_dataset/ → create_encoding_dataset.py → trained_face_dataset/
```

**Process:**
- Scans `face_dataset/` directory for person folders
- Each person folder should contain:
  - Multiple face images (`.jpg`, `.jpeg`, `.png`, `.bmp`)
  - A `metadata.json` file with person information
- Generates 128-dimensional face encodings for each image
- Stores encodings in `trained_face_dataset/` for fast recognition

**Metadata Format:**
```json
{
  "name": "John Doe",
  "id": "EMP001",
  "department": "Engineering",
  "email": "john.doe@example.com"
}
```

### 2. **Recognition Phase** (`face_recog_API.py`)

The API recognizes faces by comparing them against trained encodings:

```
Test Image → Face Encoding → Compare with Dataset → Return Best Match
```

**Recognition Process:**
1. Client sends a POST request with `subjectName` and `timestamp`
2. API constructs folder path: `{subjectName}_{timestamp}`
3. Downloads images from Supabase storage
4. Extracts face encodings from test images
5. Compares against all trained encodings using Euclidean distance
6. Returns the best match if distance ≤ tolerance (default: 0.6)

**Distance Metrics:**
- **< 0.4**: Excellent match (very confident)
- **0.4 - 0.6**: Good match (confident)
- **> 0.6**: Poor match (not recognized)

### 3. **Local Testing** (`face_recog_test_local.py`)

A standalone script for testing recognition without the API:

```bash
python face_recog_test_local.py
```

## Installation

### Prerequisites

- Python 3.8+
- CMake (required for dlib)
- Visual C++ Build Tools (Windows) or GCC (Linux/Mac)

### Step 1: Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install cmake build-essential
```

**macOS:**
```bash
brew install cmake
```

**Windows:**
- Install Visual Studio Build Tools
- Install CMake from https://cmake.org/download/

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
```
face-recognition
fastapi
uvicorn
supabase
pydantic
numpy
```

### Step 3: Set Up Supabase

1. Create a Supabase project at https://supabase.com
2. Create a storage bucket for images
3. Update credentials in `face_recog_API.py`:

```python
supabase_url = 'YOUR_SUPABASE_URL'
supabase_key = 'YOUR_SUPABASE_ANON_KEY'
bucket_name = 'YOUR_BUCKET_NAME'  # Line 163
```

**⚠️ Security Note:** For production, use environment variables instead of hardcoding credentials.

## Usage

### Step 1: Prepare Training Data

Create your face dataset:

```
face_dataset/
├── alice/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   ├── photo3.jpg
│   └── metadata.json
└── bob/
    ├── photo1.jpg
    ├── photo2.jpg
    └── metadata.json
```

### Step 2: Train the Model

```bash
python create_encoding_dataset.py
```

**Output:**
```
Processing alice...
 ✓ Encoded photo1.jpg
 ✓ Encoded photo2.jpg
 ✓ Encoded photo3.jpg
 ✓ Saved 3 encodings for alice

Processing bob...
 ✓ Encoded photo1.jpg
 ✓ Encoded photo2.jpg
 ✓ Saved 2 encodings for bob

✓ Training complete!
```

### Step 3: Test Locally (Optional)

```bash
python face_recog_test_local.py
```

### Step 4: Start the API Server

```bash
python face_recog_API.py
```

Server will start at: `http://127.0.0.1:8000`

### Step 5: Make Recognition Requests

**Upload test images to Supabase:**

Organize images in folders: `{subjectName}_{timestamp}/`

Example: `ClassroomA_2024-01-15_09-30/`

**Send POST request:**

```bash
curl -X POST "http://127.0.0.1:8000/recognize" \
  -H "Content-Type: application/json" \
  -d '{
    "subjectName": "ClassroomA",
    "timestamp": "2024-01-15_09-30"
  }'
```

**Python Example:**

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/recognize",
    json={
        "subjectName": "ClassroomA",
        "timestamp": "2024-01-15_09-30"
    }
)

result = response.json()
print(result)
```

## API Endpoints

### `GET /`

Health check endpoint.

**Response:**
```json
{
  "message": "Hello World"
}
```

### `POST /recognize`

Recognize faces from images in Supabase storage.

**Request Body:**
```json
{
  "subjectName": "string",
  "timestamp": "string"
}
```

**Response:**
```json
{
  "status": "success",
  "folder_name": "ClassroomA_2024-01-15_09-30",
  "total_images_processed": 5,
  "total_people_recognized": 3,
  "results": {
    "image_results": [
      {
        "image_path": "image1.jpg",
        "faces_found": 1,
        "person_name": "alice",
        "distance": 0.42,
        "metadata": {
          "name": "Alice Smith",
          "id": "STU001"
        }
      }
    ],
    "all_recognized_people": {
      "alice": {
        "distance": 0.42,
        "first_seen_in": "image1.jpg",
        "metadata": {...}
      }
    }
  }
}
```

### `POST /cleanup`

Clean up temporary downloaded files.

**Response:**
```json
{
  "status": "success",
  "message": "Temp files cleaned"
}
```

## Configuration

### Tolerance Adjustment

Modify the tolerance parameter to control recognition strictness:

```python
# In recognize_face() function
def recognize_face(image_path, tolerance=0.6):  # Adjust this value
```

- **Lower tolerance (0.4-0.5)**: Stricter matching, fewer false positives
- **Higher tolerance (0.6-0.7)**: More lenient, may increase false positives

## Sample Dataset

The repository includes a sample dataset for testing:

- **face_dataset/**: Contains sample face images with metadata
- **trained_face_dataset/**: Pre-generated encodings
- **test_image.jpg**: Sample test image

You can immediately test the system with:

```bash
python face_recog_test_local.py
```

## Troubleshooting

### Issue: "No module named 'face_recognition'"

**Solution:**
```bash
pip install face-recognition
```

### Issue: "CMake not found"

**Solution:** Install CMake as per Installation instructions above.

### Issue: "No face detected in the image"

**Causes:**
- Image quality too low
- Face too small in the image
- Face angle too extreme
- Poor lighting conditions

**Solutions:**
- Use high-resolution images (at least 200x200 pixels for face area)
- Ensure face is clearly visible and well-lit
- Face should be roughly frontal (±45 degrees)

### Issue: "No images found in folder"

**Causes:**
- Incorrect bucket name
- Folder doesn't exist in Supabase
- Wrong folder naming format

**Solution:** Verify Supabase storage structure matches `{subjectName}_{timestamp}` format.

## Performance Considerations

- **Training Time**: ~1-2 seconds per image
- **Recognition Time**: ~0.5-1 second per image
- **Memory Usage**: ~50MB base + ~1MB per 100 trained faces
- **Accuracy**: 99.38% on LFW benchmark (face_recognition library)

## Limitations

- Currently processes only the first detected face per image
- Requires clear, frontal face images for best results
- Does not handle face masks or significant occlusions
- Supabase credentials are hardcoded (should use environment variables in production)

## Future Enhancements

- [ ] Multi-face detection per image
- [ ] Real-time video stream recognition
- [ ] Face liveness detection
- [ ] GPU acceleration support
- [ ] Environment-based configuration
- [ ] Database integration for logging
- [ ] Authentication and rate limiting

## About EduSync

This Face Recognition API is a core component of **EduSync**, an intelligent attendance management system that automates student/employee attendance tracking using facial recognition technology. The complete EduSync platform includes:

- Automated attendance marking
- Real-time attendance analytics
- Integration with educational management systems
- Multi-location support
- Advanced reporting and insights

**For more information about the complete EduSync solution, please contact the project maintainers.**

## Contact

For questions, issues, or collaboration inquiries:
- Email: tanishraghav03@gmail.com
---

**Note:** This is an active module of an ongoing project under patent protection.


