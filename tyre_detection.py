import cv2
from ultralytics import YOLO


model = YOLO("yolov8n.pt")


video_path = "data/tyres_video.mp4"
cap = cv2.VideoCapture(video_path)


if not cap.isOpened():
    print("❌ ERROR: Could not open input video file!")
    exit()


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))


output_path = "output/tyre_output.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


if not out.isOpened():
    print("❌ ERROR: Could not create output video file!")
    exit()

print("✅ Video processing started...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("✅ Video processing complete!")
        break  


    results = model(frame)

   
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  

    
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Output video saved at: {output_path}")
