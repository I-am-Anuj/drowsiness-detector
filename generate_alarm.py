import numpy as np
import wave

sample_rate = 44100      # standard audio quality (samples per second)
duration = 1.0           # seconds
frequency = 1000          # pitch in Hz (1000 = clear beep tone)

t = np.linspace(0, duration, int(sample_rate * duration), False)
tone = np.sin(frequency * t * 2 * np.pi)

# Fade out slightly at the end so it doesn't click/pop
fade_len = int(sample_rate * 0.05)
tone[-fade_len:] *= np.linspace(1, 0, fade_len)

# Convert to 16-bit PCM format
audio = (tone * 32767).astype(np.int16)

with wave.open("alarm.wav", "w") as f:
    f.setnchannels(1)        # mono
    f.setsampwidth(2)        # 2 bytes = 16-bit
    f.setframerate(sample_rate)
    f.writeframes(audio.tobytes())

print("alarm.wav created successfully!")