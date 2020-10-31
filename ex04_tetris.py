from UniversalHandControl import *
from time import sleep, time

# Controlling the keyboard
from pynput.keyboard import Key, Controller

kbd = Controller()

def press_key(key, times=1):
    for i in range(times):
        kbd.press(key)
        kbd.release(key)

def move(event):
    event.print_line()
    rotation = event.rotation
    if -1 < rotation < -0.2:
        press_key(Key.right)
    elif 0.4 < rotation < 1.5:
        press_key(Key.left)

def rotate(event):
    event.print_line()
    press_key(Key.up)
    
def down(event): 
    event.print_line()
    press_key(Key.down)

# w = 0.5
# h = 0.3
config = {
    'app_name': 'Tetris',
    'service_connect': {'host':"localhost"},
    'ShowOptions': {'best_hands':False, 'landmarks':True},


    # 'gesture_airzones' : ['ZONE'], 
    
    'gestures' : [

        {'name': 'MOVE', 'gesture':'FIVE', 'callback': 'move', "trigger":"periodic", "first_trigger_delay":0, "next_trigger_delay": 0.2, "params": ["rotation"]},
        {'name': 'ROTATE', 'gesture':'FIST', 'callback': 'rotate', "trigger":"periodic", "first_trigger_delay":0, "next_trigger_delay": 0.4},
        {'name': 'DOWN', 'gesture':'PEACE', 'callback': 'down', "trigger":"periodic", "first_trigger_delay":0.05, "next_trigger_delay": 0.05}
    ],

    # 'airzones': [
    #     {'name': 'ZONE', 'type': 'pad', 'coordinates_type': "relative_to_anchor" , 'points': [(w/2,-h/2,0), (-w/2,-h/2, 0), (-w/2,h/2,0)], 'tolerance':0.2, 'trigger': 'continuous'},
    # ]
}


uhc = UniversalHandControl(config)
uhc.loop_forever()