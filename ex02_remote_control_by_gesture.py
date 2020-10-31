
from UniversalHandControl import *
from time import sleep, time
import logging

# RM3 Mini controller https://github.com/TheGU/rm3_mini_controller
# Used here for controlling TV
import sys
sys.path.insert(0, './rm3_mini_controller')
# from BlackBeanControl import execute_command
from RM3 import RM3
rm3 = RM3()
rm3.start()

# Sonos controller https://github.com/SoCo/SoCo
import soco
sonos = soco.discovery.any_soco()
soco_logger = logging.getLogger('soco')
soco_logger.setLevel(logging.ERROR)

# Yeelight controller https://yeelight.readthedocs.io/en/latest/
from yeelight import Bulb, discover_bulbs, Flow, transitions
yeelight_logger = logging.getLogger('yeelight')
yeelight_logger.setLevel(logging.ERROR)
bulbs = discover_bulbs()
bulb = Bulb(bulbs[0]['ip'])
bulb_last_alert_time = 0


context = "LIGHT"

def set_context(event):
    event.print_line()
    global context
    context = event.name
    uhc.say(context)
    if context == "MUSIC":
        sonos_state = sonos.get_current_transport_info()['current_transport_state']
        if sonos_state != 'PLAYING':
            sonos.play()
    elif context == "LIGHT":
        if bulb.get_properties()["power"] == "off":
            bulb.turn_on()

def command_on_off(event):
    event.print_line()
    uhc.sound("sounds/finger_click.wav")
    if context == "TV":
        rm3.send_command("on_off")
    elif context == "MUSIC":
        sonos_state = sonos.get_current_transport_info()['current_transport_state']
        if sonos_state == 'PLAYING':
            sonos.pause()
        else: # STOPPED or PAUSED_PLAYBACK
            sonos.play()
    elif context == "LIGHT":
        bulb.toggle()
        

def command_level(event):
    event.print_line()
    rotation = event.rotation
    if rotation < -0.2:
        level = "+"
    elif rotation > 0.4:
        level = "-"
    else:
        return
    if context == "TV":
        uhc.sound("sounds/tit.wav")
        rm3.send_command("vol_plus" if level == "+" else "vol_moins")
    elif context == "MUSIC":
        uhc.sound("sounds/tit.wav")
        volume = sonos.volume
        # print("Volume:", volume)
        if level == "+":
            sonos.volume = min(100, volume + 1)
        else:
            sonos.volume = max(0, volume - 1)
    elif context == "LIGHT":
        global bulb_last_alert_time
        brightness = int(bulb.get_properties()['bright'])
        if brightness == 1 and level == "-":
            now = time()
            if now - bulb_last_alert_time < 2: 
                return
            else:
                uhc.say("min")
                bulb_last_alert_time = now
            return
        elif brightness == 100 and level == "+":
            now = time()
            if now - bulb_last_alert_time < 2: 
                return
            else:
                uhc.say("max")
                bulb_last_alert_time = now
        else:
            uhc.sound("sounds/tit.wav")
        if level == "+":
            bulb.set_brightness(min(100, brightness + 20))
        else:
            bulb.set_brightness(max(0, brightness - 20))
  

def command_preset(event):
    event.print_line()
    preset = event.name
    uhc.say(preset)
    if context == "MUSIC":
        if preset == "PRESET 1":
            uri = 'x-rincon-mp3radio://http://94.23.51.96:8001/' # CINEMIX
        elif preset == "PRESET 2":
            uri = 'x-rincon-mp3radio://http://strm112.1.fm:80/bossanova_mobile_mp3' # 1.FM Bossa Nova
        elif preset == "PRESET 3":
            uri = 'x-rincon-mp3radio://http://streams.calmradio.com:6828/stream'  # Calm Radio J.S. Bach  
        sonos.play_uri(uri=uri)
    elif context == "LIGHT":
        if preset == "PRESET 1":
            rgb = (255,255,255)
            bulb.set_rgb(*rgb)
        elif preset == "PRESET 2":
            rgb = (255,0,0)   
            bulb.set_rgb(*rgb)         
        elif preset == "PRESET 3":
            rgb = (0,0,255)
            bulb.set_rgb(*rgb)
        elif preset == "PRESET 4":
            flow = Flow(count=0, transitions=transitions.disco())
            bulb.start_flow(flow)
    elif context == "TV":
        tempo = 0.6
        if preset == "PRESET 1":
            rm3.send_command(["2", "sleep 0.5", "1", "sleep 0.5", "ok"])            
        elif preset == "PRESET 2":
            rm3.send_command(["1", "sleep 0.5", "6", "sleep 0.5", "ok"])
        elif preset == "PRESET 3":
            rm3.send_command(["5", "sleep 0.5", "ok"])
                  

def command_info(event):
    event.print_line()
    if context == "MUSIC":
        track_info = sonos.get_current_track_info()
        # print(track_info)
        track_info = f"{track_info['artist']}, {track_info['title']}, {track_info['album']}"
        print(track_info)
        uhc.say(track_info)
    elif context == "TV":
        uhc.sound("sounds/shutter.wav")
        rm3.send_command("prev_channel")

def clear(event):
    import os
    os.system('clear')


config = {
    # app_name: name of the current application, used as MQTT topic name for communication with service
    'app_name': 'Remote control devices',
    # If the key 'service_connect' is absent, the service 'UniversalHandControlService' is run by the current process
    # If the key 'service_connect' is present, it must be a dictionary that contains the arguments
    # to the MQTT connect function(https://pypi.org/project/paho-mqtt/#connect-reconnect-disconnect) 
    # used to connect to the MQTT broker (can run on local or remote host). Args are host (mandatory), port, keepalive, bind_address
    # 'service_connect': {'host':"192.168.1.21"},
    'ShowOptions': { 'landmarks':True},
    'gesture_params': {"trigger": "enter", 
                        "first_trigger_delay":0.6, 
                        "next_trigger_delay":0.6, 
                        "max_moving_distance":0.1, 
                        "max_missing_frames":3},
    
    # 'airzones' : [     
    #    {'name': 'pad', 'type': 'pad', 'points': [(0.7,0.35,1.4), (-0.2,0.35,1.4), (-0.2,0.15,1.4)], 'tolerance':0.3},
    # ],

    # 'gesture_airzones': ['pad'],

    'gestures' : [

        {'name': 'LIGHT', 'gesture':'ONE', 'hand':'left', 'callback': 'set_context'},
        {'name': 'TV', 'gesture':'TWO', 'hand':'left', 'callback': 'set_context'},
        {'name': 'MUSIC', 'gesture':'THREE', 'hand':'left', 'callback': 'set_context'},
        {'name': 'ON_OFF', 'gesture':['FIST','OK'], 'hand':'right', 'callback': 'command_on_off'},
        {'name': 'LEVEL', 'gesture':'FIVE', 'hand':'right', 'callback': 'command_level', "trigger":"periodic", "first_trigger_delay":0.3, "next_trigger_delay":0.3, "params": ["rotation"]},
        {'name': 'PRESET 1', 'gesture':'ONE', 'hand':'right', 'callback': 'command_preset'},
        {'name': 'PRESET 2', 'gesture':'TWO', 'hand':'right', 'callback': 'command_preset'},
        {'name': 'PRESET 3', 'gesture':'THREE', 'hand':'right', 'callback': 'command_preset'},
        {'name': 'PRESET 4', 'gesture':'FOUR', 'hand':'right', 'callback': 'command_preset'},
        {'name': 'INFO', 'gesture':'PEACE', 'hand':'right', 'callback': 'command_info'},
        {'name': 'CLEAR', 'gesture':'PEACE', 'hand':'left', 'callback': 'clear'},
    ]
}


uhc = UniversalHandControl(config)
uhc.loop_forever()