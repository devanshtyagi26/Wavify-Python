import pyaudio
import numpy as np
import requests
import json
import time

# Audio Settings
RATE = 44100  # Sample Rate
CHUNK = 1024  # Buffer Size
BASE_FREQ = 440  # Base Frequency (A4 note)
BASE_VOLUME = 0.5  # Default Volume
BASS_CUTOFF = 500  # Default Bass Cutoff

p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=RATE,
                output=True)

def get_hand_data():
    """Fetch real-time distance values from Flask API."""
    try:
        response = requests.get("http://127.0.0.1:5000/data")
        data = json.loads(response.text)
        return data
    except Exception as e:
        print("Error fetching data:", e)
        return {"left": 0, "right": 0, "thumb_distance": 0}

def apply_low_pass_filter(samples, cutoff_freq):
    """Apply a basic low-pass filter to reduce high frequencies (simulate bass control)."""
    filtered_samples = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(len(samples), d=1.0/RATE)
    
    # Zero out frequencies above the cutoff
    filtered_samples[freqs > cutoff_freq] = 0
    return np.fft.irfft(filtered_samples).astype(np.float32)

while True:
    hand_data = get_hand_data()
    
    # Map values from hand data to audio parameters
    volume = min(max(hand_data["left"] / 200, 0), 1)  # Normalize (0 to 1)
    pitch = BASE_FREQ + (hand_data["right"] * 2)  # Adjust frequency
    bass_cutoff = max(100, BASS_CUTOFF - hand_data["thumb_distance"] * 2)  # Adjust bass filter

    # Generate sine wave with current pitch
    samples = (np.sin(2 * np.pi * np.arange(CHUNK) * pitch / RATE) * volume).astype(np.float32)

    # Apply bass control (low-pass filter)
    samples = apply_low_pass_filter(samples, bass_cutoff)

    # Stream audio
    stream.write(samples.tobytes())

    time.sleep(0.05)  # Small delay for smoother changes
