"""Text-to-speech backends with automatic fallback.

pyttsx3 alone is unreliable: on Windows it drives SAPI5 through COM, which
silently does nothing when the calling thread has no COM apartment, and on
Linux it needs espeak installed. Each backend here is independent, so the
first one that actually speaks is used and a broken one is skipped.
"""

import os
import shutil
import subprocess
import sys

# Voice ids/names containing any of these are preferred for Portuguese speech
PT_VOICE_HINTS = ('pt_br', 'pt-br', 'ptbr', 'portug', 'brazil', 'brasil',
                  'maria', 'daniel', 'helo', 'luciana')

IS_WINDOWS = sys.platform.startswith('win')


def _looks_portuguese(*fields):
    haystack = ' '.join(str(f) for f in fields if f).lower()
    return any(hint in haystack for hint in PT_VOICE_HINTS)


def _co_initialize():
    """COM must be initialized on the thread that talks to SAPI5."""
    try:
        import pythoncom
    except ImportError:
        return
    try:
        pythoncom.CoInitialize()
    except Exception:
        pass


class Pyttsx3Backend:
    name = 'pyttsx3'

    def __init__(self, rate=180):
        import pyttsx3

        _co_initialize()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)

        try:
            voices = self.engine.getProperty('voices')
        except Exception:
            voices = []
        for voice in voices:
            languages = getattr(voice, 'languages', None) or []
            if _looks_portuguese(getattr(voice, 'id', ''),
                                 getattr(voice, 'name', ''),
                                 ' '.join(str(lang) for lang in languages)):
                self.engine.setProperty('voice', voice.id)
                self.voice_name = getattr(voice, 'name', voice.id)
                break
        else:
            self.voice_name = None

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()


class Sapi5Backend:
    """Direct SAPI5 via COM, bypassing pyttsx3's event loop."""

    name = 'sapi5'

    def __init__(self, rate=180):
        if not IS_WINDOWS:
            raise RuntimeError('SAPI5 is Windows-only')

        _co_initialize()
        try:
            from win32com.client import Dispatch
        except ImportError:
            from comtypes.client import CreateObject as Dispatch

        self.voice = Dispatch('SAPI.SpVoice')
        # SAPI rate is -10..10; pyttsx3 rate is words per minute around 200
        self.voice.Rate = max(-10, min(10, int((rate - 200) / 20)))

        self.voice_name = None
        try:
            available = self.voice.GetVoices()
            for i in range(available.Count):
                token = available.Item(i)
                description = token.GetDescription()
                if _looks_portuguese(description):
                    self.voice.Voice = token
                    self.voice_name = description
                    break
        except Exception:
            pass

    def speak(self, text):
        self.voice.Speak(text)  # synchronous


class PowerShellBackend:
    """System.Speech through PowerShell — slow to start but always available."""

    name = 'powershell'

    SCRIPT = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "try {{ $s.SelectVoiceByHints('NotSet','NotSet',0,"
        "(New-Object System.Globalization.CultureInfo 'pt-BR')) }} catch {{}};"
        "$s.Speak('{text}')"
    )

    def __init__(self, rate=180):
        if not IS_WINDOWS:
            raise RuntimeError('System.Speech is Windows-only')
        self.voice_name = 'pt-BR (System.Speech)'
        self.speak('')

    def speak(self, text):
        script = self.SCRIPT.format(text=text.replace("'", "''"))
        subprocess.run(['powershell', '-NoProfile', '-Command', script],
                       check=True, capture_output=True)


class EspeakBackend:
    """espeak-ng / espeak on Linux, `say` on macOS."""

    name = 'espeak'

    def __init__(self, rate=180):
        if sys.platform == 'darwin':
            self.command = ['say', '-v', 'Luciana', '-r', str(rate)]
            self.voice_name = 'Luciana (say)'
            return
        binary = shutil.which('espeak-ng') or shutil.which('espeak')
        if not binary:
            raise RuntimeError('espeak not installed')
        self.command = [binary, '-v', 'pt-br', '-s', str(rate)]
        self.voice_name = f'pt-br ({os.path.basename(binary)})'

    def speak(self, text):
        subprocess.run(self.command + [text], check=True, capture_output=True)


BACKENDS = ([Pyttsx3Backend, Sapi5Backend, PowerShellBackend] if IS_WINDOWS
            else [Pyttsx3Backend, EspeakBackend])


def create_backend(rate=180, preferred=None):
    """Build the first usable TTS backend.

    Args:
        rate: Speech rate in words per minute.
        preferred: Backend name to try first (also read from the
                   BLINDDISTANCE_TTS environment variable).

    Returns:
        A backend exposing speak(text), or None when every backend failed.
    """
    preferred = preferred or os.environ.get('BLINDDISTANCE_TTS')
    candidates = list(BACKENDS)
    if preferred:
        candidates.sort(key=lambda b: b.name != preferred.lower())

    for backend_cls in candidates:
        try:
            backend = backend_cls(rate=rate)
        except Exception as e:
            print(f"[TTS] backend '{backend_cls.name}' unavailable: {e}")
            continue
        voice = getattr(backend, 'voice_name', None)
        if voice:
            print(f"[TTS] using {backend.name} with voice: {voice}")
        else:
            print(f"[TTS] using {backend.name}; WARNING: no Portuguese voice "
                  f"found, install a pt-BR voice for correct pronunciation")
        return backend

    print('[TTS] no working text-to-speech backend found')
    return None


if __name__ == '__main__':
    # Speaks a sample phrase with every backend so you can hear which works
    for backend_cls in BACKENDS:
        print(f"--- {backend_cls.name} ---")
        try:
            backend = backend_cls()
            print(f"    voice: {getattr(backend, 'voice_name', None)}")
            backend.speak(f'Teste de voz com {backend_cls.name}. Celular a um metro.')
            print('    ok')
        except Exception as e:
            print(f'    failed: {e}')
