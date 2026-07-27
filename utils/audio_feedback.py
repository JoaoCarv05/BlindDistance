import pyttsx3
import threading
import time
import queue
import pygame
import numpy as np

class AudioFeedback:
    def __init__(self):
        # Initialize TTS Engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 180) # Speed up speech slightly
        
        # Initialize Pygame for beeps
        pygame.mixer.init()
        
        # Threading and Queues for non-blocking audio
        self.message_queue = queue.Queue()
        self.running = True
        self.audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.audio_thread.start()
        
        # Rate limiting to prevent spamming the same warning repeatedy
        self.last_warning_time = 0.0
        self.warning_cooldown = 3.0 # Speak at most once every 3 seconds

        # Rate limiting for beeps (max 5 per second to avoid overloading pygame)
        self.last_beep_time = 0.0
        self.beep_cooldown = 0.2
        
        # We also want a generic beep for low-level proximity alerts
        # self.beep_sound = self._generate_beep() 

    def speak(self, text, force=False):
        """ Queue text to be spoken. If force=True, bypasses the cooldown limit. """
        current_time = time.time()
        
        # Always allow forced messages or messages if the cooldown has passed
        if force or (current_time - self.last_warning_time > self.warning_cooldown):
            # Drain the queue so we say the newest thing instead of lagging behind
            while not self.message_queue.empty():
                try:
                    self.message_queue.get_nowait()
                except queue.Empty:
                    break

            self.message_queue.put(text)
            self.last_warning_time = current_time

    def beep(self, frequency=800, duration_ms=200):
        """ Play a simple tone. Higher frequency or shorter duration feels more urgent. """
        now = time.time()
        if now - self.last_beep_time < self.beep_cooldown:
            return
        self.last_beep_time = now

        # Pygame mixer is non-blocking by default
        # Create a simple square wave beep mathematically (or you can load a .wav)
        sample_rate = 44100
        n_samples = int(round(duration_ms * sample_rate / 1000.0))
        
        # generate a sound
        t = np.linspace(0, duration_ms / 1000.0, n_samples, False)
        wave = np.sin(frequency * t * 2 * np.pi)
        
        # Scale to 16-bit integer for Pygame
        sound = np.int16(wave * 32767)
        
        # Needs to be 2D array (stereo)
        stereo_sound = np.empty((sound.shape[0], 2), dtype=np.int16)
        stereo_sound[:, 0] = sound
        stereo_sound[:, 1] = sound
        
        # Play it
        try:
            pg_sound = pygame.sndarray.make_sound(stereo_sound)
            pg_sound.play()
        except Exception as e:
            print(f"Audio Beep Error: {e}")

    def _audio_loop(self):
        while self.running:
            try:
                # Block until a message is received, timeout occasionally to check self.running
                msg = self.message_queue.get(timeout=0.5)
                self.tts_engine.say(msg)
                self.tts_engine.runAndWait()
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Audio Thread Error: {e}")

    def stop(self):
        self.running = False
        if self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
        pygame.mixer.quit()

# Simple test if run directly
if __name__ == '__main__':
    audio = AudioFeedback()
    audio.beep(frequency=1000)
    audio.speak("Testing audio module", force=True)
    time.sleep(2)
    audio.stop()
