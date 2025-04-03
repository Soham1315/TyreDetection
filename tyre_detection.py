import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Load input video
video_path = "data/tyres_video.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video file opened successfully
if not cap.isOpened():
    print("❌ ERROR: Could not open input video file!")
    exit()

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video writer
output_path = "output/tyre_output.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Check if the output file is opened correctly
if not out.isOpened():
    print("❌ ERROR: Could not create output video file!")
    exit()

print("✅ Video processing started...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("✅ Video processing complete!")
        break  # End of video

    # Perform YOLO detection
    results = model(frame)

    # Draw bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green rectangle

    # Write frame to output file
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Output video saved at: {output_path}")
