
import datetime 

MQTT_SERVICE_TOPIC = "_UHCService_"
ALL_GESTURES = ["ONE","TWO","THREE","FOUR","FIVE","FIST","PEACE","OK"]

class GestureConfig:
    def __init__(self, active=True, gestures=None, gesture_airzones=None):
        self.active = active
        self.gestures = gestures
        self.gesture_airzones = gesture_airzones

class AirzoneConfig:
    def __init__(self, airzones=[]):
        self.active = len(airzones) > 0
        self.airzones = airzones

class Event:
    def __init__(self, category, hand, name, callback, trigger):
        self.category = category
        self.name = name
        self.hand = hand
        self.callback = callback
        self.trigger = trigger
        self.time = datetime.datetime.now()
    def print(self):
        attrs = vars(self)
        print("--- EVENT :")
        print('\n'.join("\t%s: %s" % item for item in attrs.items()))
    def print_line(self):
        if hasattr(self, "rel_coordinates") and isinstance(self.rel_coordinates, tuple): 
            coords = f"- coords: {self.rel_coordinates[:-1]}"
        else:
            coords = ""
        print(f"{self.time.strftime('%H:%M:%S.%f')[:-3]} : {self.category} {self.name} - hand: {self.hand} - trigger: {self.trigger} - callback: {self.callback} {coords}")
        


class GestureEvent(Event):
    def __init__(self, region, gesture_entry, trigger):
        super().__init__(category = "gesture",
                    hand = region.hand if region else None,
                    name = gesture_entry["name"],
                    callback = gesture_entry["callback"],
                    trigger = trigger)
        if "params" in gesture_entry and region:
            for param in gesture_entry["params"]:
                value = getattr(region, param, None)
                setattr(self, param, value)

class AirzoneEvent(Event):
    def __init__(self, region, airzone, trigger):
        super().__init__(category="airzone",
                    hand = airzone['hand'],
                    name = airzone['name'],
                    callback = airzone['callback'],
                    trigger = trigger)
        if trigger != "leave":
            self.rel_coordinates = region.rel_coordinates
        if "params" in airzone:
            for param in airzone["params"]:
                value = getattr(region, param, None)
                setattr(self, param, value)

