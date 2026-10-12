
import cv2
import time
import queue
import threading
import subprocess
import speech_recognition as sr

from collections import Counter
from ultralytics import YOLO


# =========================================================
# SOUNDSIGHT - RASPBERRY PI 5 + USB CAMERA
# =========================================================

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

MODEL_PATH = "yolov8n_ncnn_model"
IMAGE_SIZE = 416
CONFIDENCE = 0.40

SPEAK_EVERY = 6.0
MAX_OBJECTS_TO_ANNOUNCE = 3

# Approximate typical object heights, in meters.
REAL_HEIGHTS = {
    "person": 1.70,
    "chair": 1.00,
    "couch": 0.85,
    "dining table": 0.75,
    "tv": 0.70,
    "bottle": 0.25,
    "cup": 0.12,
    "book": 0.23,
    "backpack": 0.45,
    "car": 1.50,
    "bus": 3.00,
    "truck": 2.50,
    "bicycle": 1.10,
    "motorcycle": 1.10,
    "dog": 0.60,
    "cat": 0.25,
    "potted plant": 0.50,
    "stop sign": 0.75,
}

# Starting estimate only. Calibrate for your actual camera.
FOCAL_LENGTH = 800.0

# Ignore extremely small detections for distance estimation.
MIN_PIXEL_HEIGHT = 15

# Speech and voice state.
speech_queue = queue.Queue(maxsize=5)
running = True
describe_event = threading.Event()

latest_lock = threading.Lock()
latest_details = []
latest_labels = []

last_qr_text = ""
last_announcement = ""
last_spoken_time = 0.0


# =========================================================
# SPEECH OUTPUT
# =========================================================

def speak(text, priority=False):
    """Queue speech so announcements do not overlap."""

    text = str(text).strip()
    if not text:
        return

    try:
        if priority:
            # Make room for an important announcement if necessary.
            while speech_queue.full():
                try:
                    speech_queue.get_nowait()
                    speech_queue.task_done()
                except queue.Empty:
                    break

        speech_queue.put_nowait(text)

    except queue.Full:
        pass


def speech_worker():
    """One persistent worker speaks queued messages in order."""

    while running or not speech_queue.empty():
        try:
            text = speech_queue.get(timeout=0.3)
        except queue.Empty:
            continue

        try:
            print("SPEAKING:", text)

            subprocess.run(
                ["espeak-ng", "-s", "155", text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=20
            )

        except Exception as exc:
            print("Speech output error:", exc)

        finally:
            speech_queue.task_done()


# =========================================================
# MICROPHONE SELECTION
# =========================================================

def find_microphone():
    try:
        devices = sr.Microphone.list_microphone_names()
        print(devices)
    except Exception as exc:
        print("Cannot list microphones:", exc)
        return None

    print("\nAvailable microphones:")
    for i, name in enumerate(devices):
        print(f"  {i}: {name}")

    # Prefer an obvious USB microphone.
    for i, name in enumerate(devices):
        name_lower = name.lower()
        if "usb" in name_lower or "microphone" in name_lower:
            print(f"Using microphone {i}: {name}")
            return i

    if devices:
        print(f"Using microphone 0: {devices[0]}")
        return 0

    return None


def listen_loop():
    """Listen for the spoken command 'describe'."""

    mic_index = find_microphone()

    if mic_index is None:
        print("No microphone found. Voice commands disabled.")
        return

    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True

    try:
        microphone = sr.Microphone(device_index=mic_index)

        with microphone as source:
            print("Calibrating microphone...")
            recognizer.adjust_for_ambient_noise(source, duration=1)

        print('Voice control ready. Say "describe".')

    except Exception as exc:
        print("Microphone setup failed:", exc)
        return

    while running:
        try:
            with microphone as source:
                audio = recognizer.listen(
                    source,
                    timeout=1,
                    phrase_time_limit=3
                )

            # This uses Google's online speech recognition service.
            phrase = recognizer.recognize_google(audio).lower()
            print("HEARD:", phrase)

            if "describe" in phrase:
                describe_event.set()

        except sr.WaitTimeoutError:
            continue
        except sr.UnknownValueError:
            continue
        except sr.RequestError as exc:
            print("Speech recognition network error:", exc)
            time.sleep(2)
        except Exception as exc:
            print("Microphone error:", exc)
            time.sleep(1)


# =========================================================
# DISTANCE ESTIMATION
# =========================================================

def estimate_distance(label, pixel_height):
    """
    Estimate distance using a known typical object height.

    This is approximate and depends on camera calibration,
    object orientation, and correct bounding boxes.
    """

    if label not in REAL_HEIGHTS:
        return None

    if pixel_height < MIN_PIXEL_HEIGHT:
        return None

    real_height = REAL_HEIGHTS[label]

    distance_m = (real_height * FOCAL_LENGTH) / pixel_height
    distance_ft = distance_m * 3.28084

    if distance_ft <= 0:
        return None

    return distance_ft


# =========================================================
# SCENE DESCRIPTION
# =========================================================

def make_description():
    with latest_lock:
        details = list(latest_details)
        labels = list(latest_labels)

    if not labels:
        return "I cannot identify any objects right now."

    counts = Counter(labels)
    parts = []

    for label, count in counts.most_common(6):
        if count == 1:
            parts.append(f"one {label}")
        else:
            parts.append(f"{count} {label}s")

    message = "I can see " + ", ".join(parts) + "."

    distances = [
        detail for detail in details
        if "feet away" in detail
    ]

    if distances:
        message += " " + ". ".join(distances[:3]) + "."

    return message


# =========================================================
# STARTUP
# =========================================================

def main():
    global running
    global last_qr_text
    global last_announcement
    global last_spoken_time

    print("Loading SoundSight...")

    # Start speech worker before making announcements.
    threading.Thread(
        target=speech_worker,
        daemon=True
    ).start()

    try:
        print("Loading YOLO model:", MODEL_PATH)
        model = YOLO(MODEL_PATH)
    except Exception as exc:
        print("Could not load YOLO model:", exc)
        print("Check that the NCNN model folder exists.")
        running = False
        return

    print("Opening USB camera...")

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not camera.isOpened():
        print("Cannot open camera index", CAMERA_INDEX)
        print("Check /dev/video0 or try CAMERA_INDEX = 1.")
        running = False
        return

    qr_detector = cv2.QRCodeDetector()

    threading.Thread(
        target=listen_loop,
        daemon=True
    ).start()

    speak("SoundSight started", priority=True)

    print("SoundSight running. Press Q in the camera window to quit.")

    try:
        while running:
            ok, frame = camera.read()

            if not ok:
                print("Camera frame unavailable.")
                time.sleep(0.1)
                continue

            # ---------------- QR CODE ----------------
            try:
                qr_text, qr_bbox, _ = qr_detector.detectAndDecode(frame)

                if qr_text and qr_text != last_qr_text:
                    print("QR CODE:", qr_text)
                    speak("QR code reads: " + qr_text, priority=True)
                    last_qr_text = qr_text

            except cv2.error:
                qr_bbox = None

            # ---------------- YOLO ----------------
            try:
                result = model.predict(
                    frame,
                    imgsz=IMAGE_SIZE,
                    conf=CONFIDENCE,
                    verbose=False
                )[0]
            except Exception as exc:
                print("YOLO inference error:", exc)
                continue

            details = []
            labels = []

            if result.boxes is not None:
                # Keep the highest-confidence detections.
                boxes = sorted(
                    result.boxes,
                    key=lambda box: float(box.conf[0]),
                    reverse=True
                )

                for box in boxes[:MAX_OBJECTS_TO_ANNOUNCE]:
                    class_id = int(box.cls[0])
                    label = result.names[class_id]

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    pixel_height = max(0.0, y2 - y1)

                    distance_ft = estimate_distance(
                        label, pixel_height
                    )

                    if distance_ft is not None:
                        detail = f"{label}, {distance_ft:.1f} feet away"
                    else:
                        detail = label

                    labels.append(label)
                    details.append(detail)

                    print(
                        f"DETECTED: {label}; "
                        f"box height={pixel_height:.1f}px; "
                        f"distance="
                        f"{distance_ft:.1f} ft"
                        if distance_ft is not None
                        else
                        f"DETECTED: {label}; "
                        f"box height={pixel_height:.1f}px; "
                        f"distance unavailable"
                    )

            with latest_lock:
                latest_details = list(details)
                latest_labels = list(labels)

            # ---------------- VOICE DESCRIBE ----------------
            if describe_event.is_set():
                describe_event.clear()
                description = make_description()
                print("DESCRIPTION:", description)
                speak(description, priority=True)

            # ---------------- AUTOMATIC SPEECH ----------------
            now = time.time()

            if details and now - last_spoken_time >= SPEAK_EVERY:
                # Announce distances for known objects, names otherwise.
                message = ", ".join(details) + " ahead"

                # Avoid repeating identical announcements continuously.
                if message != last_announcement:
                    print("ANNOUNCEMENT:", message)
                    speak(message)
                    last_announcement = message

                last_spoken_time = now

            # Allow a new QR announcement if the camera sees a different code.
            # The same code is announced only once until another code is seen.
            if not qr_text if 'qr_text' in locals() else True:
                pass

            # ---------------- DISPLAY ----------------
            annotated = result.plot()

            if 'qr_bbox' in locals() and qr_bbox is not None:
                try:
                    pts = qr_bbox.astype(int).reshape(-1, 2)
                    for i in range(len(pts)):
                        cv2.line(
                            annotated,
                            tuple(pts[i]),
                            tuple(pts[(i + 1) % len(pts)]),
                            (0, 255, 0),
                            2
                        )
                except Exception:
                    pass

            cv2.imshow("SoundSight - Raspberry Pi 5", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Stopping SoundSight...")

    finally:
        running = False
        camera.release()
        cv2.destroyAllWindows()
        print("SoundSight stopped.")


if __name__ == "__main__":
    main()
