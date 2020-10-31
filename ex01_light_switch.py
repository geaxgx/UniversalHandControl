from UniversalHandControl import *

# Yeelight controller https://yeelight.readthedocs.io/en/latest/
from yeelight import Bulb, discover_bulbs
bulbs = discover_bulbs()
bulb = Bulb(bulbs[0]['ip'])


def toggle_light(event):
    uhc.sound("sounds/finger_click.wav")
    event.print_line()
    bulb.toggle()



config = {
    # app_name: name of the current application (used as MQTT topic name for communication with service)
    'app_name': 'Toggle light',
    # service_connect: arguments to connect to MQTT broker 
    'service_connect': {'host':"localhost"},
    'ShowOptions': { 'landmarks':True },
    'gesture_params': {"trigger": "enter", 
                        "first_trigger_delay":0.5, 
                        "next_trigger_delay":0.5, 
                        "max_missing_frames":2},
    

    'gestures' : [
        {'name': 'ON_OFF', 'gesture':'FIST', 'hand':'right', 'callback': 'toggle_light'},
    ]
}


uhc = UniversalHandControl(config)
uhc.loop_forever()