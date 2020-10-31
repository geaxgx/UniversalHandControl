from UniversalHandControl import *
import numpy as np
import cv2
from pynput.mouse import Button, Controller
from SmoothingFilter import *

SCREEN_RES_X = 1920
SCREEN_RES_Y = 1080

def move(event):
    x,y = event.rel_coordinates
    
    e = 0.15
    p1 = SCREEN_RES_X/(1-2*e)
    q1 = -p1*e
    mx = int(max(0, min(SCREEN_RES_X-1, p1*x+q1)))
    p2 = SCREEN_RES_Y/(1-2*e)
    q2 = -p2*e
    my = int(max(0, min(SCREEN_RES_Y-1, p2*y+q2)))
    mx,my = smooth.update((mx,my))
    print(mx, my)
    mouse.position = (mx, my)
    cv2.waitKey(1)

def press_release(event):
    event.print()
    if event.trigger == "enter": 
        mouse.press(Button.left)
    elif event.trigger == "leave":
        mouse.release(Button.left)

def click(event):
    event.print()
    mouse.press(Button.left)
    mouse.release(Button.left)
        

# Pad dimensions
w = 0.5
h = 0.3

config = {
    'app_name': 'AngryBirds',
    'service_connect': {'host':"localhost"},
    'ShowOptions': {'center':True},

    'gestures' : [
        {'name': 'PRESS_RELEASE', 'gesture':['FIST', 'OK'], 'callback': 'press_release', "trigger":"enter_leave", "first_trigger_delay":0.1, "max_missing_frames":3},
        {'name': 'CLICK', 'gesture':['ONE'], 'callback': 'click', "trigger":"enter", "first_trigger_delay":0.1, "max_missing_frames":2},
    ],

    'airzones' : [     
        {'name': 'MOUSE', 'type': 'pad', 'coordinates_type': "relative_to_anchor" , 'points': [(w/2,-h/2,0), (-w/2,-h/2, 0), (-w/2,h/2,0)], 'tolerance':0.4, 'trigger': 'continuous', 'callback': 'move'},
    ]
}

mouse = Controller()
smooth = DoubleExpFilter(smoothing=0.3, prediction=0.1, jitter_radius=700, out_int=True)
# smooth.scrollbars(enable=True)
uhc = UniversalHandControl(config)
uhc.loop_forever()