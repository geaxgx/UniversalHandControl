from UniversalHandControl import *


def print_event(event):
    if event.trigger == 'periodic':
        uhc.sound("sounds/tit.wav")
    elif event.trigger == 'enter':
        uhc.sound("sounds/finger_click.wav")
    event.print_line()



config = {
    'app_name': 'DemoAirzones',
    'service_connect': {'host':"localhost"},
    'ShowOptions': {'best_hands':False, 'center':True},

    'airzone_params' : {"callback": "print_event"},
    'sensor' : {'class': 'DepthAI'},

    'airzones' : [   
        {'name': 'BUTTON', 'type': 'button', 'coordinates_type': "relative_to_anchor" , 'points': [(0.,0.,0)], 'trigger': 'enter', 'tolerance': 0.07},
        {'name': 'SLIDER', 'type': 'slider', 'coordinates_type': "relative_to_anchor" , 'points': [(0.,0.2,0), (0,-0.2,0)], 'trigger': 'periodic'},
        {'name': 'PAD', 'type': 'pad', 'coordinates_type': "relative_to_anchor" , 'points': [(0.2,-0.2,0), (-0.2,-0.2,0), (-0.2,0.2,0)], 'trigger': 'periodic'},    
    ]
}


uhc = UniversalHandControl(config) 
uhc.loop_forever()