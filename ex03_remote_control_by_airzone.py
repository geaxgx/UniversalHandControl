# remote_control_by_gesture.py

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
bulbs = discover_bulbs()
bulb = Bulb(bulbs[0]['ip'])
bulb_last_alert_time = 0
yeelight_logger = logging.getLogger('yeelight')
yeelight_logger.setLevel(logging.ERROR)


context = "MUSIC"

def light_on_off(event):
    event.print_line()
    uhc.sound("sounds/finger_click.wav")
    bulb.toggle()

def light_brightness(event):
    global bulb_last_alert_time
    event.print_line()
    brightness = int(100 * event.rel_coordinates[0])
    uhc.sound("sounds/tit.wav")
    bulb.set_brightness(brightness)
  

def music_on_off(event):
    event.print_line()
    uhc.sound("sounds/finger_click.wav")
    sonos_state = sonos.get_current_transport_info()['current_transport_state']
    if sonos_state == 'PLAYING':
        sonos.pause()
    else: # STOPPED or PAUSED_PLAYBACK
        sonos.play()

def music_volume(event):
    event.print_line()
    uhc.sound("sounds/tit.wav")
    volume = 5+int(50 * event.rel_coordinates[0])
    sonos.volume = volume


config = {
    'app_name': 'Remote control devices',
    # If the key 'service_connect' is absent, the service 'UniversalHandControlService' is run by the current process
    # If the key 'service_connect' is present, it must be a dictionary that contains the arguments
    # to the MQTT connect function(https://pypi.org/project/paho-mqtt/#connect-reconnect-disconnect) 
    # used to connect to the MQTT broker (can run on local or remote host). Args are host (mandatory), port, keepalive, bind_address
    'service_connect': {'host':"localhost"},
    'ShowOptions': {'center':True, 'best_hands':False, 'gesture':False},

    'airzone_params': {
                        "first_trigger_delay":0.1, 
                        "next_trigger_delay": 0.3,
                        "tolerance": 0.1
                        },
    
    'airzones' : [       
        {'name': 'MUSIC_ONOFF', 'type': 'button', 'points': [(-1.1, 0.26, 2.97)], 'callback': 'music_on_off'},
        {'name': 'MUSIC_VOLUME', 'type': 'slider', 'points': [(-1.33, 0.37, 2.97), (-1.31, -0.1, 3.1)], 'trigger': 'periodic', 'callback': 'music_volume'},
        {'name': 'LIGHT_ONOFF', 'type': 'button', 'points': [(-0.76, 0.27, 3)], 'callback': 'light_on_off'},
        {'name': 'LIGHT_BRIGHTNESS', 'type': 'slider', 'points': [(-0.54, 0.4, 3), (-0.56, -0.08, 3.19)], 'trigger': 'periodic', "next_trigger_delay": 0.5, 'callback': 'light_brightness'},
    ],
}


uhc = UniversalHandControl(config)
uhc.loop_forever()