import simpleaudio as sa
from queue import Queue
import threading

class SoundPlayer(threading.Thread):
    def __init__(self, sound_file_list=[], debug=False):
        """
        sound_filepath_list: list of sound files to pre-load
        """
        super().__init__()
        self.debug = debug
        self.queue = Queue()
        self.daemon = True
        # Load all sound files
        self.cache = {}
        for f in sound_file_list:
            if self.debug: print(f"{self.__class__.__name__}: loading file {f}")
            try:
                wave_obj = sa.WaveObject.from_wave_file(f)
            except:
                print(f"Error on sa.WaveObject.from_wave_file({f}) !")
                continue
            self.cache[f] = wave_obj




    def play(self, sound_file, no_cache=False):
        if self.debug: print(f"{self.__class__.__name__}: enqueue file {sound_file}")
        self.queue.put([sound_file, no_cache])

    def run(self):
        while True:
            f, no_cache = self.queue.get()
            if f in self.cache:
                wave_obj = self.cache[f]
            else:
                wave_obj = sa.WaveObject.from_wave_file(f)
                if not no_cache:
                    self.cache[f] = wave_obj
            if self.debug: print(f"{self.__class__.__name__}: playing file {f}")
            play_obj = wave_obj.play()
            play_obj.wait_done()  # Wait until sound has finished playing
            self.queue.task_done()


if __name__ == '__main__':
    from glob import glob
    from time import sleep
    dir = "sounds/*"

    file_list = glob(dir)
    sp = SoundPlayer(file_list, debug=True)
    print("Cache:", sp.cache.keys())
    sp.start()
    for f in file_list:
        sp.play(f)
        sleep(0.5)

        
    print ("now we want to exit...")
    sp.queue.join() # ends the loop when queue is empty