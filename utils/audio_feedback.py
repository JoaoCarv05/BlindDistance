import threading
import time
import queue
import pygame
import numpy as np

from utils.tts import create_backend

class AudioFeedback:
    def __init__(self, speech_rate=180, verbose=True, beep_cooldown=1.5,
                 speak_timeout=20.0):
        # The TTS backend is created inside the audio thread: on Windows (SAPI5)
        # the underlying COM object only works on the thread that created it.
        self.speech_rate = speech_rate
        self.verbose = verbose
        self.speak_timeout = speak_timeout
        self.tts_backend = None

        pygame.mixer.init()
        
        # Bounded so old announcements are dropped instead of piling up
        self.message_queue = queue.Queue(maxsize=3)
        self.running = True

        self.last_warning_time = 0.0
        self.warning_cooldown = 3.0 # Speak at most once every 3 seconds
        # Per-object cooldown: e.g., 'person' won't block 'chair' from speaking
        self._object_cooldowns = {}  # {cooldown_key: last_spoken_time}

        # Beeps are secondary to speech: keep them sparse so they don't talk over it
        self.last_beep_time = 0.0
        self.beep_cooldown = beep_cooldown
        self._speaking = False
        self._speak_started_at = None
        self._failed_backends = set()
        self._generation = 0

        # Started last so the loop never reads half-initialized state
        self._start_audio_thread()
        threading.Thread(target=self._watchdog_loop, daemon=True).start()

    def speak(self, text, force=False, cooldown_key=None):
        """ Queue text to be spoken. If force=True, bypasses the cooldown limit.
        
        Args:
            text: The text to speak.
            force: If True, bypasses cooldown entirely.
            cooldown_key: If provided, uses a per-key cooldown timer instead of
                          the global one. E.g., cooldown_key='person' won't block
                          'chair' from speaking.
        """
        current_time = time.time()
        
        if force:
            allowed = True
        elif cooldown_key:
            # Per-object cooldown: each key has its own timer
            last_time = self._object_cooldowns.get(cooldown_key, 0.0)
            allowed = (current_time - last_time > self.warning_cooldown)
            if allowed:
                self._object_cooldowns[cooldown_key] = current_time
        else:
            # Global cooldown (original behavior)
            allowed = (current_time - self.last_warning_time > self.warning_cooldown)

        if allowed:
            if not cooldown_key:
                self.last_warning_time = current_time
            
            # Drop the oldest announcement when the queue is full instead of
            # discarding everything that is still waiting to be spoken
            while True:
                try:
                    self.message_queue.put_nowait(text)
                    break
                except queue.Full:
                    try:
                        self.message_queue.get_nowait()
                    except queue.Empty:
                        pass

    def is_speaking(self):
        """ True while a message is being spoken or is still queued. """
        return self._speaking or not self.message_queue.empty()

    def beep(self, frequency=800, duration_ms=200):
        """ Play a simple tone. Higher frequency or shorter duration feels more urgent. """
        if self.is_speaking():
            return

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

    def _audio_loop(self, generation):
        # The backend is created and used on this single thread: SAPI5 COM
        # objects only work on their creating thread.
        backend = None

        while self.running and generation == self._generation:
            try:
                # Block until a message is received, timeout occasionally to check self.running
                msg = self.message_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if generation != self._generation:
                break

            if backend is None:
                backend = create_backend(rate=self.speech_rate,
                                         exclude=self._failed_backends)
                if backend is None:
                    # Every backend failed; start over rather than going mute
                    self._failed_backends.clear()
                    continue
                self.tts_backend = backend

            if self.verbose:
                print(f"[FALA] {msg}")

            self._speak_started_at = time.time()
            self._speaking = True
            try:
                backend.speak(msg)
            except Exception as e:
                print(f"Audio Thread Error: {e}")
                # A failed backend stays broken, so switch to the next one
                self._failed_backends.add(backend.name)
                backend = None
            finally:
                # A replacement loop may already own the state after a hang
                if generation == self._generation:
                    self._speaking = False
                    self._speak_started_at = None

    def _watchdog_loop(self):
        """ Replace the audio thread when an engine blocks inside speak().

        A hung engine never returns, so the thread cannot be interrupted: it is
        abandoned and a fresh loop starts with the next backend.
        """
        while self.running:
            time.sleep(1.0)
            started = self._speak_started_at
            if not started or time.time() - started < self.speak_timeout:
                continue

            stuck = getattr(self.tts_backend, 'name', None)
            print(f"[TTS] backend '{stuck}' travou apos "
                  f"{self.speak_timeout:.0f}s; trocando de motor")
            if stuck:
                self._failed_backends.add(stuck)
            self.tts_backend = None
            self._speaking = False
            self._speak_started_at = None
            self._generation += 1
            self._start_audio_thread()

    def _start_audio_thread(self):
        self.audio_thread = threading.Thread(target=self._audio_loop,
                                             args=(self._generation,),
                                             daemon=True)
        self.audio_thread.start()

    def stop(self):
        self.running = False
        if self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
        pygame.mixer.quit()

# Simple test if run directly: python -m utils.audio_feedback
if __name__ == '__main__':
    audio = AudioFeedback()
    audio.beep(frequency=1000)
    # Several phrases in a row: catches engines that only speak once
    for phrase in ("Celular a um metro e meio.",
                   "Cadeira a dois metros.",
                   "Atencao! Pessoa muito perto."):
        audio.speak(phrase, force=True)
        while audio.is_speaking():
            time.sleep(0.2)
        time.sleep(0.5)
    audio.stop()
