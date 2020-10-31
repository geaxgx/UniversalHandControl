import pyttsx3
from queue import Queue
import threading


class TTS(threading.Thread):
    def __init__(self, voice_id=None):
        super().__init__()
        # self.engine = pyttsx3.init(debug=False)
        # self.voices = [ vars(v) for v in self.engine.getProperty('voices')]  
        # if voice_id: self.set_voice(voice_id)         
        # self.engine.setProperty('rate', 150)
        self.voice_id = voice_id
        self.queue = Queue()
        self.daemon = True


    def add_say(self, msg):
        self.queue.put(msg)

    def run(self):
        while True:
            msg = self.queue.get()
            self.engine = pyttsx3.init()
            # self.voices = [ vars(v) for v in self.engine.getProperty('voices')]
            self.engine.setProperty('voice', self.voice_id)
            self.engine.setProperty('rate', 150)
            self.engine.say(msg)
            self.engine.runAndWait()
            self.queue.task_done()
            del(self.engine)




if __name__ == '__main__':
    import sys
    if sys.platform == "linux":
        voice_id = "klatt4"
    elif sys.platform == "win32":
        voice_id = 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_DAVID_11.0'
        voice_id = 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\MSTTS_V110_enGB_GeorgeM'
    tts = TTS(voice_id)
    # tts.list_voices()
    tts.start()
    tts.add_say('Sally sells seashells by the seashore.')
    tts.add_say("welcome")
        
    print ("now we want to exit...")
    # import time
    # time.sleep(10)
    tts.queue.join() # ends the loop when queue is empty