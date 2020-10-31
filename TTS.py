import pyttsx3
from queue import Queue
import threading


class TTS(threading.Thread):
    def __init__(self, voice_id=None):
        super().__init__()
        self.engine = pyttsx3.init(debug=True)
        self.voices = [ vars(v) for v in self.engine.getProperty('voices')]  
        if voice_id: self.set_voice(voice_id)         
        self.engine.setProperty('rate', 150)
        self.queue = Queue()
        self.daemon = True

    def add_say(self, msg):
        self.queue.put(msg)

    def run(self):
        while True:
            self.engine.say(self.queue.get())
            self.engine.runAndWait()
            self.queue.task_done()

    def set_voice(self, voice_id):
        if voice_id in [v['id'] for v in self.voices]:
            self.engine.setProperty('voice', voice_id) 
        else:
            print(f"The voice '{voice_id}' is not installed !")
            self.list_voices()
    def list_voices(self):
        print("Available voices:")
        for v in self.voices:
            print(v)


if __name__ == '__main__':
    import sys
    if sys.platform == "linux":
        voice_id = "klatt4"
    elif sys.platform == "win32":
        voice_id = 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_DAVID_11.0'
    tts = TTS(voice_id=voice_id)
    tts.list_voices()
    tts.start()
    tts.add_say('Sally sells seashells by the seashore.')

        
    print ("now we want to exit...")
    tts.queue.join() # ends the loop when queue is empty
 