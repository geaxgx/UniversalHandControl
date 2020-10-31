import numpy as np
import logging as log
import sys, os
import cv2
from collections import namedtuple, defaultdict
from math import ceil, sqrt, exp, pi, floor, sin, cos, atan2
from argparse import ArgumentParser
from time import time
from uhc_common import *

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
log = log.getLogger(__name__)



class Anchor:
    def __init__(self, x_center=0, y_center=0, w=0, h=0):
        self.x_center = x_center
        self.y_center = y_center
        self.w = w
        self.h = h

class HandRegion:
    def __init__(self, pd_score, pd_box, pd_kps=0):
        self.pd_score = pd_score # Palm detection score 
        self.pd_box = pd_box # Palm detection box [x, y, w, h] normalized
        self.pd_kps = pd_kps # Palm detection keypoints
        self.airzone = None
        self.gesture_in_zone = False
    def print(self):
        attrs = vars(self)
        print('\n'.join("%s: %s" % item for item in attrs.items()))

class EventHist:
    def __init__(self, triggered=False, first_triggered=False, time=0, frame_nb=0):
        self.triggered = triggered
        self.first_triggered = first_triggered
        self.time = time
        self.frame_nb = frame_nb


SSDAnchorOptions = namedtuple('SSDAnchorOptions',[
        'num_layers',
        'min_scale',
        'max_scale',
        'input_size_height',
        'input_size_width',
        'anchor_offset_x',
        'anchor_offset_y',
        'strides',
        'aspect_ratios',
        'reduce_boxes_in_lowest_layer',
        'interpolated_scale_aspect_ratio',
        'fixed_anchor_size'])

def calculate_scale(min_scale, max_scale, stride_index, num_strides):
    if num_strides == 1:
        return (min_scale + max_scale) / 2
    else:
        return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1)

def generate_anchors(options):
    """
    option : SSDAnchorOptions
    # https://github.com/google/mediapipe/blob/master/mediapipe/calculators/tflite/ssd_anchors_calculator.cc
    """
    anchors = []
    layer_id = 0
    n_strides = len(options.strides)
    while layer_id < n_strides:
        anchor_height = []
        anchor_width = []
        aspect_ratios = []
        scales = []
        # For same strides, we merge the anchors in the same order.
        last_same_stride_layer = layer_id
        while last_same_stride_layer < n_strides and \
                options.strides[last_same_stride_layer] == options.strides[layer_id]:
            scale = calculate_scale(options.min_scale, options.max_scale, last_same_stride_layer, n_strides)
            if last_same_stride_layer == 0 and options.reduce_boxes_in_lowest_layer:
                # For first layer, it can be specified to use predefined anchors.
                aspect_ratios += [1.0, 2.0, 0.5]
                scales += [0.1, scale, scale]
            else:
                aspect_ratios += options.aspect_ratios
                scales += [scale] * len(options.aspect_ratios)
                if options.interpolated_scale_aspect_ratio > 0:
                    if last_same_stride_layer == n_strides -1:
                        scale_next = 1.0
                    else:
                        scale_next = calculate_scale(options.min_scale, options.max_scale, last_same_stride_layer+1, n_strides)
                    scales.append(sqrt(scale * scale_next))
                    aspect_ratios.append(options.interpolated_scale_aspect_ratio)
            last_same_stride_layer += 1
        
        for i,r in enumerate(aspect_ratios):
            ratio_sqrts = sqrt(r)
            anchor_height.append(scales[i] / ratio_sqrts)
            anchor_width.append(scales[i] * ratio_sqrts)

        stride = options.strides[layer_id]
        feature_map_height = ceil(options.input_size_height / stride)
        feature_map_width = ceil(options.input_size_width / stride)

        for y in range(feature_map_height):
            for x in range(feature_map_width):
                for anchor_id in range(len(anchor_height)):
                    # TODO: Support specifying anchor_offset_x, anchor_offset_y.
                    x_center = (x + options.anchor_offset_x) / feature_map_width
                    y_center = (y + options.anchor_offset_y) / feature_map_height
                    new_anchor = Anchor(x_center=x_center, y_center=y_center)
                    if options.fixed_anchor_size:
                        new_anchor.w = 1.0
                        new_anchor.h = 1.0
                    else:
                        new_anchor.w = anchor_width[anchor_id]
                        new_anchor.h = anchor_height[anchor_id]
                    anchors.append(new_anchor)
        
        layer_id = last_same_stride_layer
    return anchors

def decode_bboxes(score_thresh, wi, hi, scores, bboxes, anchors):
        """
        wi, hi : NN input shape
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
        # Decodes the detection tensors generated by the model, based on
        # the SSD anchors and the specification in the options, into a vector of
        # detections. Each detection describes a detected object.
        node {
        calculator: "TfLiteTensorsToDetectionsCalculator"
        input_stream: "TENSORS:detection_tensors"
        input_side_packet: "ANCHORS:anchors"
        output_stream: "DETECTIONS:detections"
        node_options: {
            [type.googleapis.com/mediapipe.TfLiteTensorsToDetectionsCalculatorOptions] {
            num_classes: 1
            num_boxes: 2944
            num_coords: 18
            box_coord_offset: 0
            keypoint_coord_offset: 4
            num_keypoints: 7
            num_values_per_keypoint: 2
            sigmoid_score: true
            score_clipping_thresh: 100.0
            reverse_output_order: true

            x_scale: 256.0
            y_scale: 256.0
            h_scale: 256.0
            w_scale: 256.0
            min_score_thresh: 0.7
            }
        }
        }
        """
        sigmoid_scores = 1 / (1 + np.exp(-scores))
        regions = []
        for i,anchor in enumerate(anchors):
            score = sigmoid_scores[i]

            if score > score_thresh:
                # If reverse_output_order is true, sx, sy, w, h = bboxes[i,:4] 
                # Here reverse_output_order is true

                sx, sy, w, h = bboxes[i,:4]
                cx = sx * anchor.w / wi + anchor.x_center 
                cy = sy * anchor.h / hi + anchor.y_center
                w = w * anchor.w / wi
                h = h * anchor.h / hi
                box = [cx - w*0.5, cy - h*0.5, w, h]

                kps = {}
                # 0 : wrist
                # 1 : index finger joint
                # 2 : middle finger joint
                # 3 : ring finger joint
                # 4 : little finger joint
                # 5 : 
                # 6 : thumb joint
                for j, name in enumerate(["0", "1", "2", "3", "4", "5", "6"]):
                    # Here reverse_output_order is true
                    lx, ly = bboxes[i,4+j*2:6+j*2]
                    lx = lx * anchor.w / wi + anchor.x_center 
                    ly = ly * anchor.h / hi + anchor.y_center
                    kps[name] = [lx, ly]
                regions.append(HandRegion(float(score), box, kps))
        return regions

def non_max_suppression(regions, nms_thresh):

    # cv2.dnn.NMSBoxes(boxes, scores, 0, nms_thresh) needs:
    # boxes = [ [x, y, w, h], ...] with x, y, w, h of type int
    # Currently, x, y, w, h are float between 0 and 1, so we arbitrarily multiply by 1000 and cast to int
    # boxes = [r.box for r in regions]
    boxes = [ [int(x*1000) for x in r.pd_box] for r in regions]        
    scores = [r.pd_score for r in regions]
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0, nms_thresh)
    return [regions[i[0]] for i in indices]

def non_max_suppression2(regions, nms_thresh):

    if len(regions) == 0: return []

    boxes = np.array([r.pd_box for r in regions])
    # print("boxes", boxes.shape)
	# initialize the list of picked indexes
    pick = []

	# grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + x1
    y2 = boxes[:, 3] + y1

	# compute the area of the bounding boxes and grab the indexes to sort
    area = (x2 - x1) * (y2 - y1)
    idxs = [r.pd_score for r in regions]
	# sort the indexes
    idxs = np.argsort(idxs)

	# keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
		# grab the last index in the indexes list and add the index value
		# to the list of picked indexes
        i = idxs[0]
        pick.append(i)

		# find the largest (x, y) coordinates for the start of the bounding
		# box and the smallest (x, y) coordinates for the end of the bounding
		# box
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

		# compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap_areas = w * h

		# compute the ratio of overlap
        ious = overlap_areas / (area[i] + area[idxs[1:]] - overlap_areas)

        #select indices for which IOU is greather than threshold
        delete_idx = np.where(ious > nms_thresh)[0]+1
        delete_idx = np.concatenate(([0],delete_idx))
        
        #delete the above indices
        idxs = np.delete(idxs,delete_idx)

	# return only the bounding boxes that were picked
    return [regions[i] for i in pick]

def detection_letterbox_removal(regions, pad_left, pad_right, pad_top, pad_bottom):
    for r in regions:
        # r.pd_box = [x, y, w, h]
        r.pd_box[0] = (r.pd_box[0] - pad_left) / (1 - pad_left - pad_right)
        r.pd_box[1] = (r.pd_box[1] - pad_top) / (1 - pad_top - pad_bottom)
        r.pd_box[2] = r.pd_box[2] / (1 - pad_left - pad_right)
        r.pd_box[3] = r.pd_box[3] / (1 - pad_top -pad_bottom)
        # keypoints
        for k in r.pd_kps:
            r.pd_kps[k][0] = (r.pd_kps[k][0] - pad_left) /  (1 - pad_left - pad_right)
            r.pd_kps[k][1] = (r.pd_kps[k][1] - pad_top) / (1 - pad_top - pad_bottom)

def normalize_radians(angle):
    return angle - 2 * pi * floor((angle + pi) / (2 * pi))

def detections_to_rect(regions):
    # pose_detection_to_roi.pbtxt

    # # Converts each palm detection into a rectangle (normalized by image size)
    # # that encloses the palm and is rotated such that the line connecting center of
    # # the wrist and MCP of the middle finger is aligned with the Y-axis of the
    # # rectangle.
    # node {
    # calculator: "DetectionsToRectsCalculator"
    # input_stream: "DETECTIONS:palm_detections"
    # input_stream: "IMAGE_SIZE:image_size"
    # output_stream: "NORM_RECTS:palm_rects"
    # node_options: {
    #     [type.googleapis.com/mediapipe.DetectionsToRectsCalculatorOptions] {
    #     rotation_vector_start_keypoint_index: 0  # Center of wrist.
    #     rotation_vector_end_keypoint_index: 2  # MCP of middle finger.
    #     rotation_vector_target_angle_degrees: 90
    #     output_zero_rect_for_empty_detections: true
    #     }
    # }
    
    target_angle = pi * 0.5 # 90 = pi/2
    for r in regions:
        
        r.box_w = r.pd_box[2]
        r.box_h = r.pd_box[3]
        r.box_center_x = r.pd_box[0] + r.box_w / 2
        r.box_center_y = r.pd_box[1] + r.box_h / 2
        

        x0, y0 = r.pd_kps["0"] # wrist center
        x1, y1 = r.pd_kps["2"] # middle finger
        rotation = target_angle - atan2(-(y1 - y0), x1 - x0)
        r.rotation = normalize_radians(rotation)



def rotated_rect_to_points(cx, cy, w, h, rotation, wi, hi):
    b = cos(rotation) * 0.5
    a = sin(rotation) * 0.5
    points = []
    p0x = cx - a*h - b*w
    p0y = cy + b*h - a*w
    p1x = cx + a*h - b*w
    p1y = cy - b*h - a*w
    p2x = int(2*cx - p0x)
    p2y = int(2*cy - p0y)
    p3x = int(2*cx - p1x)
    p3y = int(2*cy - p1y)
    p0x, p0y, p1x, p1y = int(p0x), int(p0y), int(p1x), int(p1y)
    return [(p0x,p0y), (p1x,p1y), (p2x,p2y), (p3x,p3y)]

def rect_transformation(regions, wi, hi):
    """
    wi, hi : image input shape
    """
    # # Expands and shifts the rectangle that contains the palm so that it's likely
    # # to cover the entire hand.
    # node {
    # calculator: "RectTransformationCalculator"
    # input_stream: "NORM_RECTS:palm_rects"
    # input_stream: "IMAGE_SIZE:image_size"
    # output_stream: "hand_rects_from_palm_detections"
    # node_options: {
    #     [type.googleapis.com/mediapipe.RectTransformationCalculatorOptions] {
    #     scale_x: 2.6
    #     scale_y: 2.6
    #     shift_y: -0.5
    #     square_long: true
    #     }
    # }
    # }
    scale_x = 2.6
    scale_y = 2.6
    shift_x = 0
    shift_y = -0.5
    for r in regions:
        width = r.box_w
        height = r.box_h
        rotation = r.rotation
        if rotation == 0:
            r.rect_center_xa = (r.box_center_x + width * shift_x) * wi
            r.rect_center_ya = (r.box_center_y + height * shift_y) * hi
        else:
            x_shift = (wi * width * shift_x * cos(rotation) - hi * height * shift_y * sin(rotation)) #/ w
            y_shift = (wi * width * shift_x * sin(rotation) + hi * height * shift_y * cos(rotation)) #/ h
            r.rect_center_xa = r.box_center_x*wi + x_shift
            r.rect_center_ya = r.box_center_y*hi + y_shift

        # square_long: true
        long_side = max(width * wi, height * hi)
        # width = long_side / w
        # height = long_side / h
        r.rect_w_a = long_side * scale_x
        r.rect_h_a = long_side * scale_y
        r.rect_points = rotated_rect_to_points(r.rect_center_xa, r.rect_center_ya, r.rect_w_a, r.rect_h_a, r.rotation, wi, hi)

        

def render_rotated_rect(img, rect_points, thick_coef):
    h,w,_ = img.shape
    rect_points = np.array(rect_points)
    cv2.polylines(img, [rect_points], True, (0,255,255), int(thick_coef*2+0.5), cv2.LINE_AA)

def render_landmarks(img, region, thick_coef):
    # h,w,_ = img.shape
    src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
    dst = np.array([ (x, y) for x,y in region.rect_points[1:]], dtype=np.float32) # rect_points[0] is left bottom point !
    mat = cv2.getAffineTransform(src, dst)
    lm_xy_o = np.expand_dims(np.array([(l[0], l[1]) for l in region.landmarks]), axis=0)
    lm_xy = np.squeeze(cv2.transform(lm_xy_o, mat)).astype(np.int)


    list_connections = [[0, 1, 2, 3, 4], 
                        [0, 5, 6, 7, 8], 
                        [5, 9, 10, 11, 12],
                        [9, 13, 14 , 15, 16],
                        [13, 17],
                        [0, 17, 18, 19, 20]]

    lines = [np.array([lm_xy[point] for point in line]) for line in list_connections]
    cv2.polylines(img, lines, False, (255, 0, 0), int(thick_coef*2+0.5), cv2.LINE_AA)

    # color depending on finger state (1=open, 0=close, -1=unknown)
    color = { 1: (0,255,0), 0: (0,0,255), -1:(255,0,0)}
    radius = int(thick_coef*4+0.5)
    try:
        cv2.circle(img, (lm_xy[0][0], lm_xy[0][1]), radius, color[-1], -1)
    except :
        print(f"Exception: {sys.exc_info()} - r={radius} - xy ={lm_xy[0]}")
        print("lm_xy", lm_xy_o)
        print("lm_xy", lm_xy)
        print("dst", dst)
        print("mat", mat)
        print("lm_array", region.lm_array)
        sys.exit(1)
    

    for i in range(1,5):
        cv2.circle(img, (lm_xy[i][0], lm_xy[i][1]), radius, color[region.thumb_state], -1)
    for i in range(5,9):
        cv2.circle(img, (lm_xy[i][0], lm_xy[i][1]), radius, color[region.index_state], -1)
    for i in range(9,13):
        cv2.circle(img, (lm_xy[i][0], lm_xy[i][1]), radius, color[region.middle_state], -1)
    for i in range(13,17):
        cv2.circle(img, (lm_xy[i][0], lm_xy[i][1]), radius, color[region.ring_state], -1)
    for i in range(17,21):
        cv2.circle(img, (lm_xy[i][0], lm_xy[i][1]), radius, color[region.little_state], -1)
    # for x,y in lm_xy:
    #     cv2.circle(img, (x, y), 3, (0,128,255), -1)

def square_dist(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def distance(a, b):
    """
    a, b: 2 points in 3D (x,y,z)
    """
    return np.linalg.norm(a-b)

def get_slider_position(p, az):
    """
    p : 3D point coordinates
    az : airzone
    Test if distance from p to slider segment is less than az['tolerance']. 
    If yes return the couple (x, d) where x is the normalized position of projection of p on the slider segment and d is the normalized distance to the slider.
        x and d are floats between 0 and 1. 
    Else return None
    """
    p = np.array(p)
    ap = p-az['points'][0]
    norm_pos = np.dot(ap, az['u']/az['n_u'])
    if 0 <= norm_pos <= 1:
        d = np.linalg.norm(np.cross(ap, az['u']))
        if d <= az['tolerance']:
            return (norm_pos, d/az['tolerance'])
    return None

def get_pad_position(p, az):
    """
    p : 3D point coordinates
    az : airzone
    Test if distance from p to pad rectangle is less than az['tolerance']. 
    If yes return the triplet (x, y, d) where (x,y) is the normalized position of projection of p on the pad rectangle and d is the normalized distance to the pad.
        x and y  are floats between 0 and 1. d is a float between -1 and 1.
    Else return None
    """
    p = np.array(p)
    # import pdb
    # pdb.set_trace()
    ap = p-az['points'][0]
    d = np.dot(ap, az['w'])
    
    if abs(d) <= az['tolerance']:
        norm_x = np.dot(ap, az['u']/az['n_u'])
        norm_y = np.dot(ap, az['v']/az['n_v'])
        if 0 <= norm_x <= 1 and 0 <= norm_y <= 1:
            return (norm_x, norm_y, d/az['tolerance'])
    return None

def point_segment_distance(p, a, b):
    """
    Return distance of point p to the segment [ab] and the clamped ratio d(a,proj(p)) / d(a,b) between 0 and 1
    # https://stackoverflow.com/questions/56463412/distance-from-a-point-to-a-line-segment-in-3d-python
    """
    p = np.array(p)
    a = np.array(a)
    b = np.array(b)
    # normalized tangent vector
    n = np.linalg.norm(b - a)
    d = np.divide(b - a, n)
    # signed parallel distance components
    # s > 0 if proj(p) before a
    # t > 0 if proj(b) before p
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)
    # clamped parallel distance (0 if proj(p) between a and b)
    h = np.maximum.reduce([s, t, 0])
    # perpendicular distance component
    c = np.cross(p - a, d)
    # Clamped relative position of proj(p) on [ab] between 0 and 1
    r = max(0, min(1, -s/n))
    return np.hypot(h, np.linalg.norm(c)), r

def angle(a, b, c):
    # https://stackoverflow.com/questions/35176451/python-code-to-calculate-angle-between-three-point-using-their-3d-coordinates
    # a, b and c : points as np.array([x, y, z]) 
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)
    
class ShowOptions:
    def __init__(self,  pd_box=False, 
                        pd_kps=False,
                        center = False,
                        center_size = 7,
                        rotation=False,
                        rot_rect = False,
                        landmarks = False,
                        handedness = False,
                        gesture = True,
                        scores = False,
                        best_hands=True,
                        xyz=False,
                        airzones=True):
        self.pd_box = pd_box
        self.pd_kps = pd_kps
        self.center = center
        self.center_size = center_size
        self.rotation = rotation
        self.rot_rect = rot_rect
        self.landmarks = landmarks
        self.handedness = handedness
        self.gesture = gesture
        self.scores = scores
        self.best_hands = best_hands
        self.xyz = xyz
        self.airzones = airzones




class HandPose:
    
    def __init__(self, pd_score_thresh=0.5, pd_nms_thresh=0.3, use_landmarks=True, gesture_config=GestureConfig(), airzone_config=AirzoneConfig(), lm_score_thresh=0.9, global_score_thresh=0.7, show_options=ShowOptions(), sensor=None):
        self.pd_score_thresh = pd_score_thresh
        self.pd_nms_thresh = pd_nms_thresh
        self.use_landmarks = use_landmarks
        self.gestconf = gesture_config
        print("gestconf", self.gestconf.gestures)
        self.gesture_hist = [EventHist() for i in range(len(self.gestconf.gestures))]
        # self.active_gestures_set: set of gestures the current app is interested in (= subset of ALL_GESTURES)
        self.active_gestures_set = { g for g_entry in self.gestconf.gestures for g in g_entry['gesture']}
        self.airconf = airzone_config
        print("airconf", self.airconf.airzones)
        self.airzone_hist = [EventHist() for i in range(len(self.airconf.airzones))]
        self.lm_score_thresh = lm_score_thresh
        self.global_score_thresh = global_score_thresh
        self.show = show_options
        self.sensor = sensor
        # Stats on palm detection inference
        self.pd_infer_time_cumul = 0
        self.pd_infer_nb = 0
        # Stats on landmarks inference
        if self.use_landmarks:
            self.lm_infer_time_cumul = 0
            self.lm_infer_nb = 0
            self.lm_hand_nb = 0
        # We want to keep track of the best left hand and the best right hand (we suppose there is only one person)
        self.prev_left_hand = None
        self.prev_right_hand = None
        # Global stats
        self.frame_nb = 0
        self.global_time_cumul = 0
        
        # print('Handpose', gesture_config)
        # Create anchors
        # https://github.com/google/mediapipe/blob/master/mediapipe/graphs/hand_tracking/subgraphs/multi_hand_detection_cpu.pbtxt
        anchor_options = SSDAnchorOptions(num_layers=5, 
                                    min_scale=0.1171875,
                                    max_scale=0.75,
                                    input_size_height=256,
                                    input_size_width=256,
                                    anchor_offset_x=0.5,
                                    anchor_offset_y=0.5,
                                    strides=[8, 16, 32, 32, 32],
                                    aspect_ratios= [1.0],
                                    reduce_boxes_in_lowest_layer=False,
                                    interpolated_scale_aspect_ratio=1.0,
                                    fixed_anchor_size=True)
        self.anchors = generate_anchors(anchor_options)
        log.info(f"{len(self.anchors)} anchors have been created")

    def process(self, img, is_bgr=True):
        """
        Palm detection inference
        img : input image
        is_bgr: True for image from OpenCV (BGR), False for RGB image
        """
        self.frame_nb += 1
        self.now = time()
        self.hi, self.wi, _ = img.shape
        self.img = img
        # Padding on the small side, to get a square shape
        s = max(self.hi, self.wi)
        pad_top = pad_bottom = int((s - self.hi)/2)
        pad_left = pad_right = int((s - self.wi)/2)
        if is_bgr:
            self.img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            self.img_rgb = img
            self.img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_pad = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)
        img_pad_rgb = cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB)

        # Resize image to NN square input shape
        img_nn = cv2.resize(img_pad_rgb, (self.pd_w, self.pd_h), interpolation=cv2.INTER_AREA)

        # Normalizing padding for later (detection letterbox removal)
        pad_top = pad_bottom = pad_top / s
        pad_left = pad_right = pad_left / s
      
        # Normalize img
        img_nn = img_nn.astype(np.float32)/127.5 - 1
        # Transpose hxwx3 -> 1x3xhxw
        img_nn = np.transpose(img_nn, (2,0,1))[None,]
        
        # Palm detection inference
        pd_start_time = time()
        self.scores, self.bboxes = self.palm_detection_infer(img_nn)
        pose_infer_time = time() - pd_start_time
        # print(f"Pose infer time: {pose_infer_time*1000:.1f} ms")
        self.pd_infer_time_cumul += pose_infer_time
        self.pd_infer_nb += 1
        
        # Decode bboxes
        self.regions = decode_bboxes(self.pd_score_thresh, self.pd_w, self.pd_h, self.scores, self.bboxes, self.anchors)
        # Non maximum suppression
        self.regions = non_max_suppression(self.regions, self.pd_nms_thresh)
        # Transform back box and keypoints to original image format
        detection_letterbox_removal (self.regions, pad_left, pad_right, pad_top, pad_bottom)
        # Convert each palm detection into a rotated rectangle
        detections_to_rect(self.regions)
        
        
        # Landmarks
        if self.use_landmarks:
            # Expand and shift the rectangle that contain the palm so that it covers the entire hand
            rect_transformation(self.regions, self.wi, self.hi)
            for r in self.regions:
                img_affine = self.warp_rect_img(r.rect_points, self.img_rgb)
                # cv2.imshow("affine", img_affine)
                # Normalize
                img_nn2 = img_affine.astype(np.float32)/127.5 - 1
                # Transpose hxwx3 -> 1x3xhxw
                img_nn2 = np.transpose(img_nn2, (2,0,1))[None,]
                # Lanmarks regression inference
                lm_start_time = time()
                r.lm_score, r.handedness, lm_array = self.landmarks_regression_infer(img_nn2)
                # Bug ! Sometimes lm_array contains only huge values (like 1e+35)
                # We discard the region. Just need to check if the first entry in the array is too big
                if lm_array[0] > 1e10:
                    r.global_score = 0 # Later, regions with global_score too low are discarded
                    continue
                r.lm_array = lm_array # GX bug

                lm_infer_time = time() - lm_start_time
                self.lm_infer_time_cumul += lm_infer_time
                self.lm_infer_nb += 1
                # Global score
                r.global_score = r.pd_score * r.lm_score * (r.handedness if r.handedness > 0.5 else 1-r.handedness)
                # Keep only the region with score high enough
                # if r.lm_score < self.lm_score_thresh:
                #     r.pd_score = 0 # Later we don't process region with pd_score == 0
                #     continue
                r.hand = "right" if r.handedness > 0.5 else "left"
                r.landmarks = []
                for i in range(int(len(lm_array)/3)):
                    # x,y,z -> keep x/w,y/h,z/w (here h = w)
                    r.landmarks.append(lm_array[3*i:3*(i+1)]/self.lm_w)
                lm_z = [lm[2] for lm in r.landmarks]
            self.regions = [ r for r in self.regions if r.global_score > self.global_score_thresh ] 

        for r in self.regions:
            # Convenient to have box center coordinates in orig image, later in rendering
            r.box_center_xa = min(int(r.box_center_x * self.wi), self.wi-1)
            r.box_center_ya = min(int(r.box_center_y * self.hi), self.hi-1)

            # If depth sensor is present, get (X,Y,Z) camera coordinates for each pd_box center
            if self.sensor:
                zone_size = min(int(r.pd_box[3] * 0.2 * self.hi), 4)
                r.cam_coordinates = self.sensor.deproject(r.box_center_xa, r.box_center_ya, averaging=True, zone_size=zone_size) 
                # Check in which airzone the hand is, if any
                r.airzone, r.rel_coordinates = self.in_airzone(r.cam_coordinates)
                

        # Gesture recognition
        # if self.gestconf.active:
        self.recognize_gestures()

        # Among all the regions candidate, we want to keep the best candidates for the left hand 
        # and for the right hand
        # Selection criteria:
        # - We multiply the scores together to get a global score : pd_score * lm_score * handedness
        # and keep the regions with highest global score
        # We trust partially the handedness given by the landmarks model because it is often wrong
        if self.prev_left_hand is None or self.prev_left_hand.frame_nb - self.frame_nb > 10:
            # If it is the first time we detect a left hand 
            # or if the last time we detected a left hand was a long time ago,
            # the best candidate is chosen among those are classified as left hands
            candidates = [r for r in self.regions if r.hand == "left"]
            if candidates:
                self.left_hand = sorted(candidates, key=lambda r: r.pd_score*r.lm_score*(1-r.handedness))[-1]
                self.left_hand.active = False  # Used in rendering in case we want to only render active hands
                                                # This flag can be set to True in generate_events()
            else:
                self.left_hand = None

        if self.prev_right_hand is None or self.prev_right_hand.frame_nb - self.frame_nb > 10:
            candidates = [r for r in self.regions if r.hand == "right"]
            if candidates:
                self.right_hand = sorted(candidates, key=lambda r: r.pd_score*r.lm_score*r.handedness)[-1]
                self.right_hand.active = False  # Used in rendering in case we want to only render active hands
                                                # This flag can be set to True in generate_events()
            else:
                self.right_hand = None

        
        # Generate events
        events = self.generate_events()

        return self.regions, events

    def in_airzone(self, coords):
        """
        Return (airzone_idx, rel_coordinates)
        airzone_idx is the airzone index that includes the point defined by the absolute camera coordinates 'coords', or None if the point is not included in any airzone
        rel_coordinates is the relative coordinates of the projection onto the airzone of the point 'coords'. For a button, it is None. 
        For a slider, it is a float between 0 and 1. For a pad, it is a couple of float between 0 and 1.
        """
        for i,az in enumerate(self.airconf.airzones):
            if az['type'] == 'button':
                d = distance(coords, np.array(az['points'][0])) 
                if d < az['tolerance']:
                    return i, d/az['tolerance']
            elif az['type'] == 'slider':
                r = get_slider_position(coords, az)
                if r is not None:
                    return i, r
            elif az['type'] == 'pad':
                r = get_pad_position(coords, az)
                if r is not None:
                    return i, r
        return None, None

    def render(self):
        # Render airzones
        if self.show.airzones:
            for i,a in enumerate(self.airconf.airzones):
                if self.airzone_hist[i].frame_nb == self.frame_nb:
                    color = (0,255,0)
                else:
                    color = (230,125,215)
                if a['type'] == 'button':
                    p = a['points'][0]
                    x,y = self.sensor.project(p)
                    x2,y2 = self.sensor.project((p[0]+a['tolerance'], p[1], p[2]))
                    cv2.circle(self.img, (x,y), int(sqrt((x2-x)**2+(y2-y)**2)), color,3)
                elif a['type'] == 'slider':
                    x,y = self.sensor.project(a['points'][0])
                    x2,y2 = self.sensor.project(a['points'][1])
                    cv2.line(self.img, (x,y), (x2,y2), color, 3)
                elif a['type'] == 'pad':
                    x,y = self.sensor.project(a['points'][0])
                    x2,y2 = self.sensor.project(a['points'][1])
                    x3,y3 = self.sensor.project(a['points'][2])
                    x4,y4 = self.sensor.project(a['points'][3])
                    points = np.array([[x,y], [x2,y2], [x3,y3], [x4,y4]],dtype=np.int32)
                    cv2.polylines(self.img, [points], True, color, 3)

        for r in self.regions:  
                    
            xb,yb,w,h = r.pd_box
            # thick_coef is used to tune the thickness of the drawings wrt distance 
            thick_coef = h / 0.15
            if self.show.pd_box:
                cv2.rectangle(self.img, (int(xb*self.wi), int(yb*self.hi)), (int((xb+w)*self.wi), int((yb+h)*self.hi)),  (255,0,255), int(thick_coef * 2 + 0.5))
            if self.show.pd_kps:
                for kp_name in r.pd_kps:
                    x = int(r.pd_kps[kp_name][0] * self.wi)
                    y = int(r.pd_kps[kp_name][1] * self.hi)
                    cv2.circle(self.img, (x, y), int(thick_coef * 3 + 0.5), (0,0,255), -1)
                    cv2.putText(self.img, kp_name, (x, y+10), cv2.FONT_HERSHEY_PLAIN, 0.7*thick_coef, (0,255,0), int(thick_coef+0.5))
            if self.show.center:
                cv2.circle(self.img, (r.box_center_xa, r.box_center_ya), int(thick_coef*self.show.center_size+0.5), (255,160,0), -1)
            if self.show.rotation:
                xr = int(50 * sin(r.rotation) + r.box_center_xa)
                yr = int(-50 * cos(r.rotation) + r.box_center_ya)
                cv2.line(self.img, (r.box_center_xa, r.box_center_ya), (xr, yr), (0,255,50), int(thick_coef * 3 + 0.5), cv2.LINE_AA)
            if self.show.xyz:
                x0, y0 = int(xb*self.wi-10), int((yb+h)*self.hi)+20
                cv2.rectangle(self.img, (x0,y0), (x0+100, y0+85), (220,220,240), -1)
                cv2.putText(self.img, f"X:{r.cam_coordinates[0]*100:3.0f} cm", (x0+10, y0+20), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)
                cv2.putText(self.img, f"Y:{r.cam_coordinates[1]*100:3.0f} cm", (x0+10, y0+45), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
                cv2.putText(self.img, f"Z:{r.cam_coordinates[2]*100:3.0f} cm", (x0+10, y0+70), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
                
            if self.use_landmarks:
                if self.show.rot_rect:
                    render_rotated_rect(self.img, r.rect_points, thick_coef)
                if self.show.landmarks and r.gesture_in_zone:
                    render_landmarks(self.img, r, thick_coef)
                # if self.show.handedness:
                #     # cv2.putText(self.img, f"RIGHT {r.handedness:.2f}" if r.handedness > 0.5 else f"LEFT {1-r.handedness:.2f}", (int(xb*self.wi+10), int((yb+h)*self.hi)-20), cv2.FONT_HERSHEY_PLAIN, 1.*thick_coef, (0,255,255), 2)
                #     cv2.putText(self.img, f"RIGHT" if r.handedness > 0.5 else f"LEFT", (int(xb*self.wi+10), int((yb+h)*self.hi)-20), cv2.FONT_HERSHEY_PLAIN, 1.*thick_coef, (0,255,255), 2)

                if self.gestconf.active and self.show.gesture and r.gesture :
                    cv2.putText(self.img, r.gesture, (int(xb*self.wi+10), int((yb)*self.hi)-50), cv2.FONT_HERSHEY_PLAIN, 1.5*thick_coef, (255,255,255), 2)
                if self.show.scores:
                    cv2.putText(self.img, f"S: {r.pd_score:.2f} {r.lm_score:.2f} {r.global_score:.3f}", (int(xb*self.wi), int((yb)*self.hi)-20), cv2.FONT_HERSHEY_PLAIN, 1.*thick_coef, (0,120,255), 2)
                if self.show.best_hands:
                    if self.right_hand == r:
                        cv2.circle(self.img, (r.box_center_xa, r.box_center_ya), int(thick_coef*15), (0,240,130), -1)
                    elif self.left_hand == r:
                        cv2.circle(self.img, (r.box_center_xa, r.box_center_ya), int(thick_coef*15), (0,100,255), -1)

        
        return self.img
    
    def warp_rect_img(self, rect_points, img):
        src = np.array(rect_points[1:], dtype=np.float32) # rect_points[0] is left bottom point !
        dst = np.array([(0, 0), (self.lm_w, 0), (self.lm_w, self.lm_h)], dtype=np.float32)
        mat = cv2.getAffineTransform(src, dst)
        return cv2.warpAffine(img, mat, (self.lm_w, self.lm_h))

    def recognize_gestures(self):
        for r in self.regions:
            

            # Finger states
            # state: -1=unknown, 0=close, 1=open
            d_3_5 = distance(r.landmarks[3], r.landmarks[5])
            d_2_3 = distance(r.landmarks[2], r.landmarks[3])
            angle0 = angle(r.landmarks[0], r.landmarks[1], r.landmarks[2])
            angle1 = angle(r.landmarks[1], r.landmarks[2], r.landmarks[3])
            angle2 = angle(r.landmarks[2], r.landmarks[3], r.landmarks[4])
            r.thumb_angle = angle0+angle1+angle2
            if angle0+angle1+angle2 > 460 and d_3_5 / d_2_3 > 1.2: 
                r.thumb_state = 1
            else:
                r.thumb_state = 0

            if r.landmarks[8][1] < r.landmarks[7][1] < r.landmarks[6][1]:
                r.index_state = 1
            elif r.landmarks[6][1] < r.landmarks[8][1]:
                r.index_state = 0
            else:
                r.index_state = -1

            if r.landmarks[12][1] < r.landmarks[11][1] < r.landmarks[10][1]:
                r.middle_state = 1
            elif r.landmarks[10][1] < r.landmarks[12][1]:
                r.middle_state = 0
            else:
                r.middle_state = -1

            if r.landmarks[16][1] < r.landmarks[15][1] < r.landmarks[14][1]:
                r.ring_state = 1
            elif r.landmarks[14][1] < r.landmarks[16][1]:
                r.ring_state = 0
            else:
                r.ring_state = -1

            if r.landmarks[20][1] < r.landmarks[19][1] < r.landmarks[18][1]:
                r.little_state = 1
            elif r.landmarks[18][1] < r.landmarks[20][1]:
                r.little_state = 0
            else:
                r.little_state = -1

            # Gesture
            if self.gestconf.gesture_airzones and not(r.airzone is not None and self.airconf.airzones[r.airzone]['name'] in self.gestconf.gesture_airzones):
                r.gesture = None
                r.gesture_in_zone = False
                continue
            
            if self.gestconf.active: 
                r.gesture_in_zone = True
                if "FIVE" in self.active_gestures_set and r.thumb_state == 1 and r.index_state == 1 and r.middle_state == 1 and r.ring_state == 1 and r.little_state == 1:
                    r.gesture = "FIVE"
                elif "FIST" in self.active_gestures_set and r.thumb_state == 0 and r.index_state == 0 and r.middle_state == 0 and r.ring_state == 0 and r.little_state == 0:
                    r.gesture = "FIST"
                elif "OK" in self.active_gestures_set and r.thumb_state == 1 and r.index_state == 0 and r.middle_state == 0 and r.ring_state == 0 and r.little_state == 0:
                    r.gesture = "OK" 
                elif "PEACE" in self.active_gestures_set and r.thumb_state == 0 and r.index_state == 1 and r.middle_state == 1 and r.ring_state == 0 and r.little_state == 0:
                    r.gesture = "PEACE"
                elif "ONE" in self.active_gestures_set and r.thumb_state == 0 and r.index_state == 1 and r.middle_state == 0 and r.ring_state == 0 and r.little_state == 0:
                    r.gesture = "ONE"
                elif "TWO" in self.active_gestures_set and r.thumb_state == 1 and r.index_state == 1 and r.middle_state == 0 and r.ring_state == 0 and r.little_state == 0:
                    r.gesture = "TWO"
                elif "THREE" in self.active_gestures_set and r.thumb_state == 1 and r.index_state == 1 and r.middle_state == 1 and r.ring_state == 0 and r.little_state == 0:
                    r.gesture = "THREE"
                elif "FOUR" in self.active_gestures_set and r.thumb_state == 0 and r.index_state == 1 and r.middle_state == 1 and r.ring_state == 1 and r.little_state == 1:
                    r.gesture = "FOUR"
                else:
                    r.gesture = None
            
            
            
    def generate_events(self):
        events = []

        airzone_event_added = None
        for r in [self.right_hand, self.left_hand]: 
            
            # Gesture events 
            if self.gestconf.active:
                for i,g in enumerate(self.gestconf.gestures):
                    h = self.gesture_hist[i]
                    trigger = g['trigger']
                    if r and r.gesture \
                        and (r.hand == g['hand'] or g['hand'] == 'any') \
                        and r.gesture in g['gesture'] :                       
                        if trigger == "continuous":
                            events.append(GestureEvent(r, g, "continuous"))
                        else: # trigger in ["enter", "enter_leave", "periodic"]:
                            if not h.triggered:
                                if h.time != 0 and (self.frame_nb - h.frame_nb <= g['max_missing_frames']):
                                    if  h.time and ((h.first_triggered and self.now - h.time > g['next_trigger_delay']) or (not h.first_triggered and self.now - h.time > g['first_trigger_delay'])):
                                        
                                        if trigger == "enter" or trigger == "enter_leave":
                                            h.triggered = True
                                            events.append(GestureEvent(r, g, "enter"))
                                        else: # "periodic"
                                            h.time = self.now
                                            h.first_triggered = True
                                            events.append(GestureEvent(r, g, "periodic"))
                                        
                                else:
                                    h.time = self.now
                                    h.first_triggered = False
                            else:
                                if self.frame_nb - h.frame_nb > g['max_missing_frames']:
                                    h.time = self.now
                                    h.triggered = False
                                    h.first_triggered = False
                                    if trigger == "enter_leave":
                                        events.append(GestureEvent(r, g, "leave"))
                        h.frame_nb = self.frame_nb
                
                    else:
                        if h.triggered and self.frame_nb - h.frame_nb > g['max_missing_frames']:
                            h.time = self.now
                            h.triggered = False
                            h.first_triggered = False 
                            if trigger == "enter_leave":
                                events.append(GestureEvent(r, g, "leave"))                  
        
            # Airzone events
            if self.airconf.active:
                for i,az in enumerate(self.airconf.airzones):
                    h = self.airzone_hist[i]
                    trigger = az['trigger']
                    if r and r.airzone is not None and r.airzone == i:
                        if trigger == "continuous":
                            if i != airzone_event_added:
                                events.append(AirzoneEvent(r, az, "continuous"))
                                airzone_event_added = i
                        else: # trigger in ["enter", "enter_leave", "periodic", "periodic_leave"]
                            if not h.triggered:
                                if h.time != 0 and (self.frame_nb - h.frame_nb <= az['max_missing_frames']):
                                    if  h.time and ((h.first_triggered and self.now - h.time > az['next_trigger_delay']) or (not h.first_triggered and self.now - h.time > az['first_trigger_delay'])):
                                        if trigger == "enter" or trigger == "enter_leave":
                                            h.triggered = True
                                            events.append(AirzoneEvent(r, az, "enter"))
                                        else: # periodic or periodic_leave
                                            h.time = self.now
                                            h.first_triggered = True
                                            events.append(AirzoneEvent(r, az, "periodic"))
                                else:
                                    h.time = self.now
                                    h.first_triggered = False
                            else:
                                if self.frame_nb - h.frame_nb > az['max_missing_frames']:
                                    h.time = self.now
                                    h.triggered = False
                                    h.first_triggered = False
                                    if trigger == "enter_leave" or trigger == "periodic_leave":
                                        events.append(AirzoneEvent(r, az, "leave"))
                        h.frame_nb = self.frame_nb
                    else:
                        if (h.triggered or h.first_triggered) and self.frame_nb - h.frame_nb > az['max_missing_frames']:
                            h.time = self.now
                            h.triggered = False
                            h.first_triggered = False 
                            if trigger == "enter_leave" or trigger == "periodic_leave":
                                events.append(AirzoneEvent(r, az, "leave")) 

        return events






    def print_stats(self):
        log.info(f"Average palm detection inference time : {self.pd_infer_time_cumul/self.pd_infer_nb*1000:.1f} ms ({self.pd_infer_nb} inferences)")
        if self.use_landmarks: log.info(f"Average landmarks regr inference time : {self.lm_infer_time_cumul/self.lm_infer_nb*1000:.1f} ms ({self.lm_infer_nb} inferences)")
        # log.info(f"Global frame processing time          : {self.global_time_cumul/self.frame_nb*1000:.1f} ms ({self.frame_nb} frames)")


PALM_DETECTION_OPENVINO_FP16 = "models/256x256/palm_detection_builtin/FP16/palm_detection_builtin.xml" 
LANDMARK_OPENVINO_FP16 = "models/256x256/hand_landmark_new/FP16/hand_landmark_new.xml"
PALM_DETECTION_OPENVINO_FP32 = "models/256x256/palm_detection_builtin/FP32/palm_detection_builtin.xml" 
LANDMARK_OPENVINO_FP32 = "models/256x256/hand_landmark_new/FP32/hand_landmark_new.xml"


class HandPoseOpenvino(HandPose):
    def __init__(self, pd_xml=PALM_DETECTION_OPENVINO_FP32, 
                lm_xml=LANDMARK_OPENVINO_FP32, 
                pd_device="CPU", 
                lm_device="CPU", 
                **kwargs):
    
        """
        pd_xml : palm detection model xml filepath
        pd_device : device on which palm detection model is run
        lm_xml : landmark model xml filepath
        lm_device : device on which landmark model is run
        kwargs : args inherited from HandPose class
        """
        from openvino.inference_engine import IENetwork, IECore
        super().__init__(**kwargs)

        log.info("Loading Inference Engine")
        self.ie = IECore()
        log.info("Device info:")
        versions = self.ie.get_versions(pd_device)
        log.info("{}{}".format(" "*8, pd_device))
        log.info("{}MKLDNNPlugin version ......... {}.{}".format(" "*8, versions[pd_device].major, versions[pd_device].minor))
        log.info("{}Build ........... {}".format(" "*8, versions[pd_device].build_number))

        # Palm detection model
        self.pd_name = os.path.splitext(pd_xml)[0]
        pd_bin = self.pd_name + '.bin'
        log.info("Palm detection model - Loading network files:\n\t{}\n\t{}".format(pd_xml, pd_bin))
        self.pd_net = IENetwork(model=pd_xml, weights=pd_bin)
        # [ INFO ] Input blob: input - shape: [1, 3, 256, 256]
        # [ INFO ] Output blob: classificators - shape: [1, 2944, 1] : tensor scores
        # [ INFO ] Output blob: regressors - shape: [1, 2944, 18] : tensor bboxes
        self.pd_input_blob = next(iter(self.pd_net.inputs))
        log.info(f"Input blob: {self.pd_input_blob} - shape: {self.pd_net.inputs[self.pd_input_blob].shape}")
        _,_,self.pd_h,self.pd_w = self.pd_net.inputs[self.pd_input_blob].shape
        for o in self.pd_net.outputs.keys():
            log.info(f"Output blob: {o} - shape: {self.pd_net.outputs[o].shape}")
            self.pd_scores = "classificators"
            self.pd_bboxes = "regressors"
        log.info("Loading palm detection model to the plugin")
        self.pd_exec_net = self.ie.load_network(network=self.pd_net, num_requests=1, device_name=pd_device)

        # Landmarks model
        if self.use_landmarks:
            if lm_device != pd_device:
                log.info("Device info:")
                versions = self.ie.get_versions(pd_device)
                log.info("{}{}".format(" "*8, pd_device))
                log.info("{}MKLDNNPlugin version ......... {}.{}".format(" "*8, versions[pd_device].major, versions[pd_device].minor))
                log.info("{}Build ........... {}".format(" "*8, versions[pd_device].build_number))

            self.lm_name = os.path.splitext(lm_xml)[0]
            lm_bin = self.lm_name + '.bin'
            log.info("Landmark model - Loading network files:\n\t{}\n\t{}".format(lm_xml, lm_bin))
            self.lm_net = IENetwork(model=lm_xml, weights=lm_bin)
            # [ INFO ] Input blob: input - shape: [1, 3, 256, 256]
            # [ INFO ] Output blob: StatefulPartitionedCall/functional_1/tf_op_layer_Sigmoid/Sigmoid - shape: [1, 1, 1, 1]: score
            # [ INFO ] Output blob: StatefulPartitionedCall/functional_1/tf_op_layer_Sigmoid_1/Sigmoid_1 - shape: [1, 1, 1, 1] : handedness
            # [ INFO ] Output blob: StatefulPartitionedCall/functional_1/tf_op_layer_ld_21_3d/ld_21_3d - shape: [1, 63] : landmarks
                
            self.lm_input_blob = next(iter(self.lm_net.inputs))
            log.info(f"Input blob: {self.lm_input_blob} - shape: {self.lm_net.inputs[self.lm_input_blob].shape}")
            _,_,self.lm_h,self.lm_w = self.lm_net.inputs[self.lm_input_blob].shape
            for o in self.lm_net.outputs.keys():
                log.info(f"Output blob: {o} - shape: {self.lm_net.outputs[o].shape}")
                self.lm_score = "StatefulPartitionedCall/functional_1/tf_op_layer_Sigmoid/Sigmoid"
                self.lm_handedness = "StatefulPartitionedCall/functional_1/tf_op_layer_Sigmoid_1/Sigmoid_1"
                self.lm_landmarks = "StatefulPartitionedCall/functional_1/tf_op_layer_ld_21_3d/ld_21_3d"
            log.info("Loading landmark model to the plugin")
            self.lm_exec_net = self.ie.load_network(network=self.lm_net, num_requests=1, device_name=lm_device)

    def palm_detection_infer(self, img_nn):
        """
        img_nn: normalized and resized image, can be directly fed into the palm detection model
        """
        inference_result = self.pd_exec_net.infer(inputs={self.pd_input_blob: img_nn})
        scores = np.squeeze(inference_result[self.pd_scores])
        bboxes = inference_result[self.pd_bboxes][0]
        return scores, bboxes

    def landmarks_regression_infer(self, img_nn):
        """
        img_nn: normalized and resized image, can be directly fed into the landmarks model
        Returns :
        - lm_score : probability between 0 and 1 that img_nn contains a hand
        - lm handedness : probability between 0 and 1 that the hand is a right hand
        - lm_array : array of size 63 (=3x21) containing 21 (x, y, z) coordinates of 21 landmarks
        """
        inference_result = self.lm_exec_net.infer(inputs={self.lm_input_blob: img_nn})
        lm_score  = np.squeeze(inference_result[self.lm_score])
        handedness = np.squeeze(inference_result[self.lm_handedness])
        lm_array = np.squeeze(inference_result[self.lm_landmarks])
        return lm_score, handedness, lm_array

if __name__ == '__main__':
    

    parser = ArgumentParser()
    parser.add_argument("-i", "--input", default='0', type=str,
                            help="Path to image or video file or webcam (default=%(default)s)")
    # parser.add_argument("--pd_m", default=PALM_DETECTION_MODEL, type=str,
    #                         help="Required. Path to an .xml file for palm detection model (dfault=%(default)s)")
    # parser.add_argument("--pd_d", default='CPU', type=str,
    #                         help="Target device to infer on the palm detection model (dfault=%(default)s)")                        
    # parser.add_argument("--lm_m", default=LANDMARK_MODEL, type=str,
    #                         help="Required. Path to an .xml file for landmark model (dfault=%(default)s)")
    # parser.add_argument("--lm_d", default='CPU', type=str,
    #                         help="Target device to infer on the landmark regression model (dfault=%(default)s)")                        
    # parser.add_argument("--lm_off", action="store_true",
    #                         help="Disable landmarks regression (keep only palm detection")
    # parser.add_argument("--lm_2", action="store_true",
    #                         help="Set batch size of landmarks model to 2 (instead of 1)")                        

    args = parser.parse_args()
    if args.input == '0':
        image_mode = False
        cam=cv2.VideoCapture(0)
    elif args.input.endswith('.jpg') or args.input.endswith('.png') :
        image_mode = True
        img = cv2.imread(args.input)
        first = True
    else:
        image_mode = False
        input_stream = args.input
        cam=cv2.VideoCapture(args.input)

    show_options = ShowOptions(pd_box=True)
    
    hp = HandPoseOpenvino(
            pd_xml=PALM_DETECTION_OPENVINO_FP32,
            lm_xml=LANDMARK_OPENVINO_FP32,
            pd_score_thresh=0.9,
            pd_nms_thresh=0.1,
            use_landmarks=True, 
            lm_score_thresh=0.9,
            show_options=show_options
    )

    
    while True:
        if not image_mode:
            ret,frame=cam.read()
            if not ret: break
            frame_orig = frame.copy()
        else:
            frame = img.copy()
        hp.process(frame)
        resframe = hp.render()
        cv2.imshow("Handpose", resframe)

        if image_mode and first:
            print("Saving output in output_HandPose.jpg")
            cv2.imwrite("output_HandPose.jpg", resframe)
            first = False
        k = cv2.waitKey(1)
        if k == 27:
            break
        elif k == 32:
            # Pause
            k = cv2.waitKey(0)
            if k == ord('s'): # Save picture
                print("Saving image in imgsav.jpg")
                cv2.imwrite("imgsav.jpg", frame_orig)
        elif k == ord('1'):
            hp.show.pd_box = not hp.show.pd_box
        elif k == ord('2'):
            hp.show.pd_kps = not hp.show.pd_kps
        elif k == ord('3'):
            hp.show.center = not hp.show.center
        elif k == ord('4'):
            hp.show.rot_rect = not hp.show.rot_rect
        elif k == ord('5'):
            hp.show.rotation = not hp.show.rotation
        elif k == ord('6'):
            hp.show.landmarks = not hp.show.landmarks
        elif k == ord('7'):
            hp.show.gesture = not hp.show.gesture
        elif k == ord('8'):
            hp.show.scores = not hp.show.scores
        elif k == ord('9'):
            hp.show.xyz = not hp.show.xyz
        elif k == ord('0'):
            hp.show.best_hands = not hp.show.best_hands


    cv2.destroyAllWindows()
    hp.print_stats()