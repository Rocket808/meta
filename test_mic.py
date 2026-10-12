import speech_recognition as sr
import audioop

MIC_INDEX = 1  # Your 8MP USB Camera microphone
DURATION = 5   # Record for 5 seconds

print("=== SoundSight Microphone Test ===")
print(f"Testing microphone {MIC_INDEX}...")
print("Speak normally for the next 5 seconds!")

try:
    with sr.Microphone(device_index=MIC_INDEX) as source:
        recognizer = sr.Recognizer()

        # Record audio for a fixed duration
        audio = recognizer.record(source, duration=DURATION)

    # Measure the loudness of the recorded audio
    raw_audio = audio.get_raw_data()
    rms = audioop.rms(raw_audio, 2)

    print("\nRecording finished.")
    print(f"Audio volume (RMS): {rms}")

    if rms > 100:
        print("SUCCESS: Microphone is capturing sound!")
    else:
        print("WARNING: Very little sound detected.")
        print("Check the microphone, mute setting, and device index.")

except Exception as e:
    print(f"\nMICROPHONE TEST FAILED: {e}")
