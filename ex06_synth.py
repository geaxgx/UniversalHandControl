from UniversalHandControl import *
import numpy as np
import cv2
import rtmidi
from rtmidi.midiconstants import CONTROL_CHANGE, CHANNEL_VOLUME
import time

midiout = rtmidi.MidiOut()
available_ports = midiout.get_ports()
print(available_ports)


if available_ports:
    midiout.open_port(0)


nb_keys = 13
key_pressed = [False] * nb_keys


def note_on_all_channels(note, velocity=112):
    print("Note ON:", note, velocity)
    for ch in range(nb_channels):
        note_on = [0x90|ch, note, velocity]
        midiout.send_message(note_on)

def note_off_all_channels(note):
    print("Note OFF:", note)
    for ch in range(nb_channels):
        note_off = [0x80|ch, note, 0]
        midiout.send_message(note_off)

def keyboard(event): # enter_leave
    
    if event.trigger == "periodic":

        is_white_key = event.rel_coordinates[1] > 0.5
        if is_white_key:
            i = int(event.rel_coordinates[0] * 8)
            if i < 3:
                key = i*2
            elif i < 7:
                key = i*2 -1
            else: # i == 8
                key = 12
        else:
            i = int(event.rel_coordinates[0] * 16) 
            if 1 <= i <= 4:
                key = 1 + int((i-1)/2) * 2
            elif 7 <= i <= 12:
                key = 8 + int((i-9)/2) * 2
            else:
                key = None
        print("white", is_white_key, key, i)
        
        if key is not None and not key_pressed[key]: # We are just arriving on this key 
            # Off the current played note if any
            for i,pressed in enumerate(key_pressed):
                if pressed:
                    note_off_all_channels(key)
                    key_pressed[i] = False
            velocity = int(128-(event.rel_coordinates[2] +1) * 64)
            note_on_all_channels(48+key, velocity)
            key_pressed[key] = True
    elif event.trigger == "leave" :
        for i,pressed in enumerate(key_pressed):
                # if pressed:
                note_off_all_channels(48+i)
                key_pressed[i] = False

def volume(event):
    volume = int(event.rel_coordinates[0]*127)
    set_volume(channel, volume)
    

def set_volume(channel, volume):
    print(f"Set volume channel {channel+1} to {volume}")
    midiout.send_message([0xb0|channel, 0x07, volume & 0x7F])


def set_channel(event):
    global channel
    if event.name == 'CHANNEL 1':
        channel = 0
    elif event.name == 'CHANNEL 2':
        channel = 1
    elif event.name == 'CHANNEL 3':
        channel = 2
    uhc.say("Channel "+str(channel+1))
    print(f"Select channel {channel+1}")
    

w = 0.7
h = 0.3
config = {
 'app_name': 'Music',
 'service_connect': {'host':"192.168.1.21"},
 'ShowOptions': {'center':True, 'gesture':True, 'best_hands':False},
 'HandPose': {'lm_score_thresh':0.7, 'pd_nms_thresh':0.1, 'pd_score_thresh':0.4, 'global_score_thresh':0.7},
 'airzone_params': {
 "first_trigger_delay":0.1, 
 "next_trigger_delay": 0.1,
 "max_missing_frames":3,
 "tolerance": 0.1
 },

'gestures' : [
        {'name': 'CHANNEL 1', 'gesture':'ONE', 'hand':'any', 'callback': 'set_channel'},
        {'name': 'CHANNEL 2', 'gesture':'TWO', 'hand':'any', 'callback': 'set_channel'},
        {'name': 'CHANNEL 3', 'gesture':'THREE', 'hand':'any', 'callback': 'set_channel'},
],

'airzones' : [ 
 {'name': 'keyboard', 'type': 'pad', 'coordinates_type': "relative_to_anchor" , 'points': [(w/2,-h/2,-0.), (-w/2,-h/2, -0.), (-w/2,h/2,-0.)], "tolerance": 0.1, 'trigger': 'periodic_leave', "callback": "keyboard"},
 {'name': 'volume', 'type': 'slider', 'coordinates_type': "relative_to_anchor" , 'points': [(0.,0,0), (-0.5,0,0.)], 'trigger': 'periodic', "callback": "volume"},
 ]
}

# for i,pressed in enumerate(key_pressed):
#     print("Key off:", i)
#     note_off = [0x80, 48 + i, 0]
#     midiout.send_message(note_off)


uhc = UniversalHandControl(config)
nb_channels = 3
channel = 1
set_volume(0, 0)
set_volume(2, 0)
uhc.loop_forever()