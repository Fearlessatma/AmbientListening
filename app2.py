from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel
import threading
import keyboard
import pyaudio
import wave
import speech_recognition as sr
from openai import OpenAI
import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
OUTPUT_FILENAME = "continuous_recording.wav"
TRANSCRIPTION_FILE = "transcription_summary.txt"
SUMMARY_FILE = "summary.txt"
stop_recording = False

# def listen_for_stop():
#     global stop_recording
#     keyboard.wait("s")
#     stop_recording = True

def record_audio(app_instance):
    global stop_recording
    stop_recording = False
    app_instance.update_status("Recording started...")
    # threading.Thread(target=listen_for_stop, daemon=True).start()
    
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    
    while not stop_recording:
        data = stream.read(CHUNK)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    with wave.open(OUTPUT_FILENAME, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))
    
    app_instance.update_status("Recording ended.")
    transcribed_text = transcribe_audio(OUTPUT_FILENAME)
    if transcribed_text:
        summary_text = generate_summary()
        app_instance.update_summary(summary_text)

def transcribe_audio(audio_file_path):
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
        transcribed_text = transcription.text
        with open(TRANSCRIPTION_FILE, "w", encoding="utf-8") as f:
            f.write(transcribed_text + "\n\n")
        return transcribed_text
    except Exception as e:
        return f"Error during transcription: {e}"

def generate_summary():
    try:
        with open(TRANSCRIPTION_FILE, "r", encoding="utf-8") as f:
            transcribed_text = f.read().strip()
        
        if transcribed_text:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI medical assistant that summarizes conversations and provides proper information about diseases in few words."
                    },
                    {
                        "role": "user",
                        "content": f"Summarize the following conversation which should be paragraph format and provide what could be the possible reasons for the symptoms mentioned by the patients in few words:\n\n{transcribed_text}"
                    }
                ],
                max_tokens=150
            )
            summary_text = response.choices[0].message.content.strip()
            with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
                f.write(summary_text)
            return summary_text
        return "No transcription available."
    except Exception as e:
        return f"Error generating summary: {e}"
    
    

class AmbientListening(MDApp):
    def build(self):
        self.layout = MDBoxLayout(orientation='vertical', padding=20, spacing=20)
        
        self.status_label = MDLabel(text="Press start to record.", halign='center')
        self.start_btn = MDRaisedButton(text="Start Recording", on_release=self.start_recording,pos_hint={"center_x": 0.5, "center_y": 0.5})
        self.stop_btn = MDRaisedButton(text="Stop Recording", on_release=self.stop_recording,pos_hint={"center_x": 0.5, "center_y": 0.5})
        
        self.summary_label = MDLabel(text="Summary will appear here.", halign='center')
        
        self.layout.add_widget(self.status_label)
        self.layout.add_widget(self.start_btn)
        self.layout.add_widget(self.stop_btn)
        self.layout.add_widget(self.summary_label)
        
        return self.layout
    
    def start_recording(self, instance):
        threading.Thread(target=record_audio, args=(self,), daemon=True).start()
    
    def stop_recording(self, instance):
        global stop_recording
        stop_recording = True
    
    def update_summary(self, summary_text):
        self.summary_label.text = summary_text
    
    def update_status(self, status_text):
        self.status_label.text = status_text

if __name__ == "__main__":
    AmbientListening().run()