import sys
import jsonpickle
if sys.platform == "linux":
    from TTS import TTS
elif sys.platform == "win32":
    from TTS_win import TTS
from SoundPlayer import SoundPlayer
from uhc_common import MQTT_SERVICE_TOPIC




"""
UniversalHandControl
client side
"""

# Default values for config parameters
# Each one of these parameters can be superseded by a new value if specified in client code
DEFAULT_CONFIG = {
    'gesture_params': 
        {
        "callback": "_DEFAULT_",
        "hand": "any",
        "trigger": "enter", 
        "first_trigger_delay":0.8, 
        "next_trigger_delay":0.8, 
        "max_moving_distance":0.2, 
        "max_missing_frames":3,
        "params": []
        },

    'airzone_params':
        {
        "callback": "_DEFAULT_",
        "hand": "any",
        "trigger": "enter", 
        "first_trigger_delay":0.3, 
        "next_trigger_delay":0.3, 
        "max_missing_frames":3,
        "tolerance": 0.1,
        "coordinates_type": "absolute", # "absolute" or "relative_to_anchor"
        "airzone_move": "translation"  # "translation" or "rotation" ("rotation" not yet available)
        },

    'gesture_airzones' : None,

    'sensor' : {'class':'Realsense'}, 
    'HandPose': {'lm_score_thresh':0.9, 'pd_nms_thresh':0.1},
}

DEFAULT_SENSOR_ARGS = {'Realsense': {'res_w':640, 'res_h':480}, 'DepthAI': {}}

anchor_coordinates = None

def _process_events(events, app_globals):
    for e in events:
        # e.print()
        if e.callback == "_DEFAULT_":
            default_callback(e)
        else:
            app_globals[e.callback](e)

# The callback for when the client receives a CONNACK response from the server.
def UHC_on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    app_name = userdata.app_name
    client.subscribe(app_name)

def UHC_on_message(client, userdata, message):
    msg = jsonpickle.decode(message.payload)
    # print("Message received " , msg)
    # print("message topic=",message.topic)
    # print("message qos=",message.qos)
    # print("message retain flag=",message.retain)

    if msg['type'] == "EXIT":
        print(f"Message received on topic '{message.topic}'")
        print("App exit asked by UHCS service")
        client.unsubscribe(userdata.app_name)
        client.disconnect()
        sys.exit(0)
    elif msg['type'] == "EVENTS":
        events = msg['events']
        if events: _process_events(events, userdata.app_globals)



def merge_config(c1, c2):
    res = {}
    for k1,v1 in c1.items():
        if k1 in c2:
            if isinstance(v1, dict):
                assert isinstance(c2[k1], dict), f"{c2[k1]} should be a dictionary"
                res[k1] = merge_config(v1, c2[k1])
            else:
                res[k1] = c2[k1]
        else:
            res[k1] = v1
    for k2,v2 in c2.items():
        if k2 not in c1:
            res[k2] = v2
    return res
            
class UniversalHandControl:
    def __init__(self, config, set_will_message=True, use_tts=True):
        global anchor_coordinates
        # Check if UHCService is run by the current process (no key 'service' in config dict) 
        # or by a server process (that can be on a remote computer, 'service':{'ip':"1.20.1.2", 'port':5555})
        self.config = merge_config(DEFAULT_CONFIG, config)
        # Sensor args
        if 'args' not in self.config['sensor'] or not isinstance(self.config['sensor']['args'], dict):
            self.config['sensor']['args'] = DEFAULT_SENSOR_ARGS[self.config['sensor']['class']]
        else:
            self.config['sensor']['args'] = merge_config(DEFAULT_SENSOR_ARGS[self.config['sensor']['class']], self.config['sensor']['args'])
        self.local = 'service_connect' not in self.config
        self.use_tts = use_tts
        # Initialize TTS
        if use_tts:
            if sys.platform == "linux":
                voice_id = None #"klatt4"
            elif sys.platform == "win32":
                voice_id = 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_DAVID_11.0'
            self.tts = TTS(voice_id=voice_id)
            self.tts.start()
        # Check if there are airzones with point coordinates are relative to an anchor.
        # If yes, we first need to ask the user to interactively defined the position of these anchor,
        # in order to replace relative by absolute corrdinates 
        if "airzones" in self.config:
            for a in self.config['airzones']:
                if a.get("coordinates_type", self.config['airzone_params']['coordinates_type']) == 'relative_to_anchor':
                    anchor_coordinates = None
                    self.say(f"Setting of the {a['type']} called {a['name']}.. Please place your hand at the position you want and keep it still for a few seconds.")
                    ask_user_for_anchor_position(a['name'], a['type'], self.config.get('service_connect', None), self.config['sensor'])
                    self.say("Thank you")
                    a['coordinates_type'] = 'absolute'
                    dx, dy, dz = anchor_coordinates
                    new_points = []
                    for x, y, z in a['points']:
                        if a.get("airzone_move", self.config['airzone_params']['airzone_move']) == "translation":
                            xa, ya, za = x+dx, y+dy, z+dz 
                            new_points.append((xa,ya,za))
                    a['points'] = new_points

        self.app_globals = sys._getframe(1).f_globals # Used to be access and run the callbacks defined in the calling app

        if self.local:
            from UniversalHandControlService import UniversalHandControlService
            self.uhcs = UniversalHandControlService(self.config)
        else:
            # We access UHCService via mqtt
            import paho.mqtt.client as mqtt
            assert "app_name" in self.config, "There is no key 'app_name' defined in config"
            self.app_name = config["app_name"]
            assert isinstance(config['service_connect'], dict) and 'host' in config['service_connect'], "Value of 'service_connect' in config must be a dict and contain 'host'"

            connect_args = config['service_connect']
            self.client = mqtt.Client(client_id=self.app_name, userdata=self)
            self.client.on_connect = UHC_on_connect
            self.client.on_message = UHC_on_message
            # Will message will be send by the broker on behalf of the app,
            # if the app disconnects ungracefully
            if set_will_message:
                self.client.will_set(MQTT_SERVICE_TOPIC, payload=jsonpickle.encode({'type':'EXIT', 'name':self.app_name}), qos=0, retain=False)
            self.client.connect(**connect_args)

            # Send config to server in INIT message
            msg = {"type":"INIT", "config": self.config}
            self.client.publish(MQTT_SERVICE_TOPIC, jsonpickle.encode(msg)) 

        

        # Initialize SounPlayer
        self.soundplayer = SoundPlayer()
        self.soundplayer.start()

        print("UHC: init finished")
    
    def say(self, msg):
        if self.use_tts:
            self.tts.add_say(msg)
    
    def sound(self, file):
        self.soundplayer.play(file)

    def loop_forever(self):
        if self.local:
            running = True
            while running:
                running,events = self.uhcs.loop()
                if events: _process_events(events, self.app_globals)
        else:
            self.client.loop_forever() # The events are received by UHC_on_message callback

    def loop(self):
        if self.local:
            _, events = self.uhcs.loop()
            if events: _process_events(events, self.app_globals)
        else:
            self.client.loop()

    def exit(self):
        if not self.local:
            print("Exiting - Sending EXIT message to service")
            self.client.publish(MQTT_SERVICE_TOPIC, jsonpickle.encode({'type': 'EXIT', 'name':self.app_name}))

def default_callback(event):
    event.print()

def set_anchor_position(event):
    global anchor_coordinates
    event.print()
    anchor_coordinates = event.cam_coordinates


def ask_user_for_anchor_position(az_name, az_type, service_connect, sensor_config):
    config = {
        'ShowOptions': { 'center':True, 'best_hands': False, 'gesture': False},

        'gestures' : [
        {'name': 'XYZ', 'gesture':'ALL', 'callback': 'set_anchor_position', "trigger":"enter", "first_trigger_delay":3,  "params": ["cam_coordinates"]},   
        ]
    }
    config['app_name'] = "ANCHOR_XYZ_" + az_name
    config['sensor'] = sensor_config
    if service_connect:
        config['service_connect'] = service_connect
    print("ask_user_for_anchor_position", config)
    _uhc = UniversalHandControl(config, set_will_message=False, use_tts=False)
    print("anchor_coordinates",anchor_coordinates)
    while anchor_coordinates == None:
        _uhc.loop()
    _uhc.exit()
    

if __name__ == '__main__':

    config = {
        # app_name: name of the current application, used as MQTT topic name for communication with service
        'app_name': 'Test',
        # If the key 'service_connect' is absent, the service 'UniversalHandControlService' is run by the current process
        # If the key 'service_connect' is present, it must be a dictionary that contains the arguments
        # to the MQTT connect function(https://pypi.org/project/paho-mqtt/#connect-reconnect-disconnect) 
        # used to connect to the MQTT broker (can run on local or remote host). Args are host (mandatory), port, keepalive, bind_address
        # 'service_connect': {'host':"localhost"},
        'ShowOptions': {'landmarks': True},

        'gestures' : [
            {'name': 'GESTURE', 'gesture':'ALL', 'trigger': 'enter'}
        ],
    }

    uhc = UniversalHandControl(config)
    uhc.loop_forever()

   