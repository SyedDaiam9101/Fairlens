"""
Detectify Dataset Tool
Captures images from webcam and prepares a YOLOv8 dataset.
"""
import cv2
import time
import os
import yaml
from pathlib import Path

def create_dataset_structure():
    """Create folder structure."""
    dirs = [
        "dataset/images/train", 
        "dataset/images/val", 
        "dataset/labels/train", 
        "dataset/labels/val"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return Path("dataset")

def capture_faces(name, class_id, num_images=50):
    """Capture faces from webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print(f"\n--- CAPTURING: {name} ---")
    print("Press 'c' to start capturing, 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow("Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret: break
        
        # Save image
        timestamp = int(time.time() * 1000)
        filename = f"{name}_{timestamp}.jpg"
        img_path = f"dataset/images/train/{filename}"
        cv2.imwrite(img_path, frame)
        
        # Create YOLO Label (Full image as face for simplicity in this helper, 
        # ideally user would crop, but for rapid prototyping we assume face is valid)
        # Better approach: We just capture raw images now, and user labels them?
        # OR: We use a face detector (Haar/dlib) to auto-crop?
        # Let's use a simple Center Crop assumption or just save empty label so user can label?
        # Actually, for "Zero to Hero", let's assume the user is the main subject.
        # We will create a label file assuming the object is in the center 50% of screen.
        
        h, w, _ = frame.shape
        # x_center, y_center, width, height (normalized)
        label_line = f"{class_id} 0.5 0.5 0.6 0.8" 
        
        lbl_path = f"dataset/labels/train/{name}_{timestamp}.txt"
        with open(lbl_path, "w") as f:
            f.write(label_line)
            
        # Copy to val occasionally (20% split)
        if count % 5 == 0:
            cv2.imwrite(f"dataset/images/val/{filename}", frame)
            with open(f"dataset/labels/val/{name}_{timestamp}.txt", "w") as f:
                f.write(label_line)

        # Visual feedback
        cv2.putText(frame, f"Captured: {count}/{num_images}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (int(w*0.2), int(h*0.1)), (int(w*0.8), int(h*0.9)), (0, 255, 0), 2)
        cv2.imshow("Capture", frame)
        cv2.waitKey(100) # Delay for variety
        count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {count} images for {name}.")

def create_data_yaml(names):
    """Generate data.yaml for YOLO."""
    data = {
        "path": str(Path("dataset").absolute()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(names)}
    }
    
    with open("data.yaml", "w") as f:
        yaml.dump(data, f)
    print("\nCreated data.yaml")

def main():
    create_dataset_structure()
    
    people = []
    print("Enter names of people to train (type 'done' to finish):")
    while True:
        name = input("Name: ").strip()
        if name.lower() == 'done' or name == "":
            break
        people.append(name)
    
    if not people:
        print("No names entered.")
        return

    for i, name in enumerate(people):
        capture_faces(name, i)
    
    create_data_yaml(people)
    
    print("\nDataset ready!")
    print("To train, run: python -m detectify train data.yaml")

if __name__ == "__main__":
    main()
