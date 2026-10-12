from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import cv2
import os
import time
import threading
import speech_recognition as sr

print("Loading models...")
yolo_model = YOLO("yolov8n.pt")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

cap = cv2.VideoCapture(0)

last_time = 0
speak_every = 3
describe_requested = False
last_qr_text = ""

REAL_HEIGHTS = {
    "person": 1.7,
    "chair": 1.0,
    "tv": 0.7
}

FOCAL_LENGTH = 800

# QR code
qr_detector = cv2.QRCodeDetector()


#  MIC 
def listen_loop():
    global describe_requested

    recognizer = sr.Recognizer()
    mic = sr.Microphone(device_index=1)

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)

    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source, phrase_time_limit=3)

            text = recognizer.recognize_google(audio).lower()
            print("Heard:", text)

            if "describe" in text:
                describe_requested = True

        except:
            pass


threading.Thread(target=listen_loop, daemon=True).start()


while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # QR CODE READING
    qr_data, bbox, _ = qr_detector.detectAndDecode(frame)

    if qr_data and qr_data != last_qr_text:
        print("QR:", qr_data)
        os.system(f'say "Entering {qr_data}"')
        last_qr_text = qr_data

    # YOLO + DISTANCE
    results = yolo_model(frame, verbose=False)
    boxes = results[0].boxes
    names = results[0].names

    current_time = time.time()

    if boxes is not None and len(boxes) > 0:
        labels = []

        for box in boxes[:3]:
            cls = int(box.cls[0])
            label = names[cls]

            y1 = box.xyxy[0][1].item()
            y2 = box.xyxy[0][3].item()
            pixel_height = y2 - y1

            if label in REAL_HEIGHTS and pixel_height > 0:
                distance_m = (REAL_HEIGHTS[label] * FOCAL_LENGTH) / pixel_height
                distance_ft = distance_m * 3.281
                labels.append(f"{label} {distance_ft:.1f} feet")
            else:
                labels.append(label)

        labels = list(dict.fromkeys(labels))
        text = ", ".join(labels) + " ahead"

        if current_time - last_time > speak_every:
            print("YOLO:", text)
            os.system(f'say "{text}"')
            last_time = time.time()

    # BLIP ON VOICE COMMAND
    if describe_requested:
        print("Voice requested scene description")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        inputs = processor(images=image, return_tensors="pt")
        out = blip_model.generate(**inputs, max_new_tokens=20)

        caption = processor.decode(out[0], skip_special_tokens=True)

        print("BLIP:", caption)
        os.system(f'say "{caption}"')

        describe_requested = False

    # qr box
    if bbox is not None:
        pts = bbox.astype(int).reshape(-1, 2)
        for i in range(len(pts)):
            cv2.line(
                frame,
                tuple(pts[i]),
                tuple(pts[(i + 1) % len(pts)]),
                (0, 255, 0),
                2
            )

    annotated_frame = results[0].plot()
    cv2.imshow("SoundSight Voice Mode", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
