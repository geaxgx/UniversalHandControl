from HandPose import *

import cv2
import jsonpickle
from uhc_common import *

"""
UniversalHandControl
client side
"""

# Uncomment 2 following lines when doing screen recording
# cv2.namedWindow("Universal Hand Control")
# cv2.moveWindow("Universal Hand Control", 0, 0)


def merge_dicts(d1, d2):
    return {**d1, **d2}


def check_keys(dic, mandatory_keys):
    for k in mandatory_keys:

        assert k in dic.keys(), f"Mandatory key '{k}' not present in {dic}"

class UniversalHandControlService:
    def __init__(self, config):
        # self.local is True when UniversalHandControlService is instantiated by UniversalHandControl.py 
        # on the same computer. In that case, UniversalHandControlService returns the events to UniversalHandControl.
        # self.local is False when UniversalHandControlService acts as a service communicating via MQTT.
        # In that case, UniversalHandControlService publishes the events 
        self.local = __name__ != "__main__"
        self.config = config
        self.parse_config()
        show_options = ShowOptions(**self.show_options_args)
        if self.sensor_params['class'] == "Realsense":
            from RealsenseSensor import RealsenseSensor
            self.sensor = RealsenseSensor(**self.sensor_params['args'])
        elif self.sensor_params['class'] == "DepthAI":
            from DepthAISensor import DepthAISensor
            self.sensor = DepthAISensor(**self.sensor_params['args'])
        self.hp = HandPoseOpenvino(show_options=show_options, gesture_config=self.gesture_config, airzone_config=self.airzone_config, sensor= self.sensor, **self.handpose_args)
        self.show_step =0

    def parse_config(self):
        self.show_options_args = self.config.get('ShowOptions',{})
        self.handpose_args = self.config.get('HandPose', {})
        self.sensor_params = self.config.get('sensor', {})
        # Airzones parsing
        self.parse_airzones()
        # Gestures parsing
        self.parse_gestures()

    def parse_airzones(self):
        mandatory_keys = ['name', 'type', 'points']
        optional_keys = self.config['airzone_params'].keys()
        airzones = []
        for a in self.config.get('airzones',[]):
            a['points'] = [np.array(p) for p in a['points']]
            check_keys(a, mandatory_keys)
            mandatory_args = { k:a[k] for k in mandatory_keys}

            optional_args = {k:a.get(k, self.config['airzone_params'][k]) for k in optional_keys}
            all_args = merge_dicts(mandatory_args, optional_args)
            # Precomputing some variables used later when checking is a point is inside an airzone
            if a['type'] == 'slider':
                p1,p2 = a['points']
                n_u = np.linalg.norm(p2 - p1) # Norme p1p2
                u = (p2 -p1)/n_u # Unit vector on segment [p1p2]
                all_args['n_u'] = n_u
                all_args['u'] = u
            elif a['type'] == 'pad':
                p1,p2,p3 = a['points']
                # We consider that p1p2 is on our first axis u on the plan (p1p2p3)
                # Due to imprecision of measures, p2p3 is not precisely orthogonal to p1p2
                # So we first determine the third axis w that is // to p1p2^p2p3
                # Then we can get the second axis v = w^u
                n_u = np.linalg.norm(p2 - p1) # Norme p1p2
                u = (p2 - p1)/n_u # u: unit vector on segment [p1p2]
                w = np.cross(p3 - p2, u); w = w / np.linalg.norm(w) # w: unit vector perpendicular to plan (p1p2p3)
                v = np.cross(w, u) # v: unit vector perpendicular to u and w, should be "almost" // to p2p3
                n_v = np.dot(p3-p2,v) # Norme de la projection de p2p3 sur w
                all_args['n_u'] = n_u
                all_args['u'] = u
                all_args['n_v'] = n_v
                all_args['v'] = v
                all_args['w'] = w
                # We rectify p3
                a['points'][2] = p2 + n_v * v
                # ... and add p4 (used to render the rectangle)
                a['points'].append(p1 + n_v * v)
            airzones.append(all_args)
        self.airzone_config = AirzoneConfig(airzones)

    def parse_gestures(self):
        # Are there airzones associated to gestures (ie gestures has to happen into these airzones to be valid) ?
        if self.config['gesture_airzones'] is not None:
            assert isinstance(self.config['gesture_airzones'], list) and self.airzone_config.active
            # Check that airzones listed in config['gesture_airzones'] are defined into config
            defined_airzones_name = [ az['name'] for az in self.config['airzones']]
            for az_name in self.config['gesture_airzones']:
                assert az_name in defined_airzones_name

        mandatory_keys = ['name', 'gesture']
        optional_keys = self.config['gesture_params'].keys()
        gestures = []
        if 'gestures' in self.config:
            for g in self.config['gestures']:
                check_keys(g, mandatory_keys)
                gesture = g['gesture']
                if isinstance(gesture, list):
                    for x in gesture:
                        assert x in ALL_GESTURES, f"Incorrect gesture {x} in {g} !"
                elif gesture == 'ALL':
                    g['gesture'] = ALL_GESTURES
                else:
                    # 'gesture' is a simple gesture. Transform it into a list
                    assert gesture in ALL_GESTURES, f"Incorrect gesture {gesture} in {g} !"
                    g['gesture'] = [gesture]
                optional_args = {k:g.get(k, self.config['gesture_params'][k]) for k in optional_keys}
                mandatory_args = { k:g[k] for k in mandatory_keys}
                all_args = merge_dicts(mandatory_args, optional_args)
                gestures.append(all_args)
            self.gesture_config = GestureConfig(True, gestures, self.config['gesture_airzones'])
        else:
            self.gesture_config = GestureConfig(False, [], None)     
    
    def loop(self):
        # This function is one step of a loop
        # So it is typically called in outer while loop
        frames = self.sensor.next_frames()
        color = frames['color'].copy()
        regions, events = self.hp.process(frames['color'])

        if self.show_options_args:
            annotated_frame = self.hp.render()
            cv2.imshow("Universal Hand Control", annotated_frame)
            k = cv2.waitKey(1)
            if k == 27:
                self.terminate()
                return False, events
            elif k == 32:
                # Pause
                k = cv2.waitKey(0)
                if k == ord('s'): # Save picture
                    print("Saving image in imgsav.jpg")
                    cv2.imwrite("imgsav.jpg", color)
            
            elif k == ord('0'):
                self.hp.show.airzones = not self.hp.show.airzones
            
            elif k == ord('1'):
                self.hp.show.pd_box = not self.hp.show.pd_box
            elif k == ord('2'):
                self.hp.show.pd_kps = not self.hp.show.pd_kps
            elif k == ord('3'):
                self.hp.show.center = not self.hp.show.center
            elif k == ord('4'):
                self.hp.show.rot_rect = not self.hp.show.rot_rect
            elif k == ord('5'):
                self.hp.show.rotation = not self.hp.show.rotation
            elif k == ord('6'):
                self.hp.show.landmarks = not self.hp.show.landmarks
            elif k == ord('7'):
                self.hp.show.gesture = not self.hp.show.gesture
            elif k == ord('8'):
                self.hp.show.scores = not self.hp.show.scores
            elif k == ord('9'):
                self.hp.show.xyz = not self.hp.show.xyz
            elif k == ord('b'):
                self.hp.show.best_hands = not self.hp.show.best_hands
            elif k == ord("d"):
                if self.show_step == 0:
                    self.hp.show.pd_box = self.hp.show.pd_kps = self.hp.show.center = self.hp.show.rot_rect = self.hp.show.rotation = self.hp.show.landmarks = self.hp.show.gesture = self.hp.show.scores = self.hp.show.xyz = self.hp.show.best_hands = False
                elif self.show_step == 1:
                    self.hp.show.pd_box = self.hp.show.pd_kps = True 
                    self.hp.show.center = self.hp.show.rot_rect = self.hp.show.rotation = self.hp.show.landmarks = self.hp.show.gesture = self.hp.show.scores = self.hp.show.xyz = self.hp.show.best_hands = False
                elif self.show_step ==  2:
                    self.hp.show.pd_box = self.hp.show.pd_kps = self.hp.show.center = True
                    self.hp.show.rot_rect = self.hp.show.rotation = self.hp.show.landmarks = self.hp.show.gesture = self.hp.show.scores = self.hp.show.xyz = self.hp.show.best_hands = False
                elif self.show_step ==  3:
                    self.hp.show.pd_box = self.hp.show.pd_kps = self.hp.show.center = self.hp.show.rotation = True
                    self.hp.show.rot_rect =  self.hp.show.landmarks = self.hp.show.gesture = self.hp.show.scores = self.hp.show.xyz = self.hp.show.best_hands = False
                elif self.show_step ==  4:
                    self.hp.show.pd_box = self.hp.show.pd_kps = self.hp.show.center = self.hp.show.rotation = self.hp.show.rot_rect = True
                    self.hp.show.landmarks = self.hp.show.gesture = self.hp.show.scores = self.hp.show.xyz = self.hp.show.best_hands = False
                elif self.show_step ==  5:
                    self.hp.show.rot_rect = True
                    self.hp.show.pd_box = self.hp.show.pd_kps = self.hp.show.center = self.hp.show.rotation =  self.hp.show.landmarks = self.hp.show.gesture = self.hp.show.scores = self.hp.show.xyz = self.hp.show.best_hands = False
                elif self.show_step ==  6:
                    self.hp.show.rot_rect = self.hp.show.landmarks= True
                    self.hp.show.pd_box = self.hp.show.pd_kps = self.hp.show.center = self.hp.show.rotation = self.hp.show.gesture = self.hp.show.scores = self.hp.show.xyz = self.hp.show.best_hands = False
                elif self.show_step ==  7:
                    self.hp.show.landmarks = self.hp.show.best_hands = True
                    self.hp.show.rot_rect = self.hp.show.pd_box = self.hp.show.pd_kps = self.hp.show.center = self.hp.show.rotation = self.hp.show.gesture = self.hp.show.scores = self.hp.show.xyz =  False
                elif self.show_step ==  8:
                    self.hp.show.landmarks = self.hp.show.gesture = True
                    self.hp.show.rot_rect = self.hp.show.pd_box = self.hp.show.pd_kps = self.hp.show.center = self.hp.show.rotation = self.hp.show.best_hands =  self.hp.show.scores = self.hp.show.xyz =  False
                elif self.show_step ==  9:
                    self.hp.show.xyz = self.hp.show.center = True
                    self.hp.show.pd_box = self.hp.show.pd_kps = self.hp.show.rot_rect = self.hp.show.rotation = self.hp.show.landmarks = self.hp.show.gesture = self.hp.show.scores = self.hp.show.best_hands = False
                self.show_step = (self.show_step + 1) %10
        

        if events:
            if self.local:
                return True, events
            else:
                client.publish(app, jsonpickle.encode({'type': 'EVENTS', 'events': events}))

        return True, None
    
    def terminate(self):
        print("UHCS instance terminating")
        self.sensor.stop()
        # cv2.destroyAllWindows()
        self.hp.print_stats()

            
# The callback for when the client receives a CONNACK response from the server.
def UHCS_on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(MQTT_SERVICE_TOPIC)


def UHCS_on_message(client, userdata, message):
    global app, uhcs
    # print("message received " ,message.payload.decode("utf-8"))
    # print("message topic=",message.topic)
    # print("message qos=",message.qos)
    # print("message retain flag=",message.retain)
    if message.topic == MQTT_SERVICE_TOPIC:
        msg = jsonpickle.decode(message.payload)
        print("Message received", msg)
        if msg['type'] == "INIT":
            config = msg['config']
            new_app = config["app_name"]
            print("app:", app, "new_app:", new_app)
            if new_app != app:
                if app is not None:
                    uhcs.terminate()
                    # End communication with current client 
                    print(f"Ending communication with {app}")
                    client.publish(app, jsonpickle.encode({'type': 'EXIT'}))

            app = new_app
             # Will message will be send by the broker on behalf of the app,
            # if the app disconnects ungracefully
            client.will_set(app, payload=jsonpickle.encode({'type': 'EXIT'}), qos=0, retain=False)
            # A reconnect is needed to take into account the Will message
            client.reconnect()
            uhcs = UniversalHandControlService(config)
        elif msg['type'] == "EXIT":
            print(f"UHCS terminating asked by app '{msg['name']}'")
            uhcs.terminate()
            app = None

if __name__ == "__main__":
    # Subscribing to MQTT_SERVICE_TOPIC topic
    # and wait for config message from application 
    import paho.mqtt.client as mqtt

    client = mqtt.Client()
    client.on_connect = UHCS_on_connect
    client.on_message = UHCS_on_message
    
    client.connect("localhost")
    app = None
    uhcs = None

    running = True
    while running:
        client.loop(timeout=0.001)
        if app is not None:
            running,_ = uhcs.loop()

    

  