import pyaudio
import numpy as np
import datetime
import os
import time

print("CRIMEVIS AI - Audio Scream Detection Active...")

os.makedirs("incidents", exist_ok=True)

# Audio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
SCREAM_THRESHOLD = 3000  # Volume level for scream
SCREAM_DURATION = 1  # seconds of loud sound = alert

p = pyaudio.PyAudio()
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

print("Listening for screams and loud sounds...")
print("Press Ctrl+C to stop")

scream_start = None
last_alert = None

try:
    while True:
        # Read audio chunk
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)

        # Get volume level
        volume = np.abs(audio_data).mean()
        now = datetime.datetime.now()

        # Visual volume bar
        bar_length = int(volume / 100)
        bar = "█" * min(bar_length, 50)
        print(f"\rVolume: {volume:.0f} |{bar:<50}|", end="")

        if volume > SCREAM_THRESHOLD:
            if scream_start is None:
                scream_start = time.time()
                print(f"\n⚠️  LOUD SOUND DETECTED! Volume: {volume:.0f}")
            else:
                duration = time.time() - scream_start
                if duration >= SCREAM_DURATION:
                    print(f"\n🚨 SCREAM/DISTRESS DETECTED!")
                    print(f"Duration: {duration:.1f}s | Volume: {volume:.0f}")

                    if last_alert is None or (now - last_alert).seconds >= 30:
                        timestamp = now.strftime('%Y%m%d_%H%M%S')
                        log_file = f"incidents/audio_{timestamp}.txt"
                        with open(log_file, 'w') as f:
                            f.write(f"AUDIO ALERT\n")
                            f.write(f"Time: {now}\n")
                            f.write(f"Volume: {volume:.0f}\n")
                            f.write(f"Duration: {duration:.1f}s\n")
                            f.write(f"Type: SCREAM/DISTRESS\n")
                        print(f"INCIDENT LOGGED: {log_file}")
                        last_alert = now
        else:
            scream_start = None

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n\nStopping audio detection...")
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Audio detection stopped.")
