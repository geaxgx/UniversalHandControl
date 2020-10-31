import cv2                                
import numpy as np                
import os
import json

import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
   

class RealsenseSensor:
    def __init__(self, res_w=1280, res_h=720, fps=30, use_color=True, use_depth=True, show_depth=True, preset=1, color_scheme=0):
        assert use_color or use_depth, "At least 1 of the 2 flags 'use_color' and 'use_depth' must be set to True"

        self.use_color = use_color
        self.use_depth = use_depth
        self.use_color_depth = use_color and use_depth
        self.show_depth = use_depth and show_depth
        # Resolution in pixels
        self.res_w = res_w
        self.res_h = res_h
        # Preset: 1=default, 2=hand, 3=high accuracy,

        # Setup of the sensor
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        if self.use_color_depth: self.align = rs.align(rs.stream.color)

        # Color stream
        if self.use_color: self.cfg.enable_stream(rs.stream.color, res_w, res_h, rs.format.bgr8, 30)
        # Depth stream to configuration
        # Format z16: 0<=z<=65535
        if self.use_depth: self.cfg.enable_stream(rs.stream.depth, res_w, res_h, rs.format.z16, fps)
        self.queue = rs.frame_queue(2 if self.use_color_depth else 1) # With a buffer capacity of 2, we reduce latency
        self.pipe_profile = self.pipe.start(self.cfg, self.queue)
        
        # Color scheme: 0: Jet, 1: Classic, 2: WhiteToBlack, 3: BlackToWhite, 4: Bio, 5: Cold, 6: Warm, 7: Quantized, 8: Pattern, 9: Hue
        if self.show_depth:
            self.colorizer = rs.colorizer()
            self.colorizer.set_option(rs.option.color_scheme, color_scheme)

        if self.use_depth:
            self.dev = self.pipe_profile.get_device()
            # self.advnc_mode = rs.rs400_advanced_mode(self.dev)
            # print("Advanced mode is", "enabled" if self.advnc_mode.is_enabled() else "disabled")
            self.depth_sensor = self.dev.first_depth_sensor()
            # Set preset
            self.preset = preset
            self.depth_sensor.set_option(rs.option.visual_preset, self.preset)
            print(f"Current preset: {self.depth_sensor.get_option_value_description(rs.option.visual_preset, self.depth_sensor.get_option(rs.option.visual_preset))}")

        # # Set the disparity shift     
        # if disparity_shift != -1:
        #     self.depth_table_control_group = self.advnc_mode.get_depth_table()
        #     self.depth_table_control_group.disparityShift = disparity_shift
        #     self.advnc_mode.set_depth_table(self.depth_table_control_group)
        # self.depth_table_control_group = self.advnc_mode.get_depth_table()
        # print("Disparity shift:", self.depth_table_control_group.disparityShift)

        # Set depth units
        # self.set_depth_units(depth_units)

            # Get intrinsics
            self.stream_profile = self.pipe_profile.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
            self.intrinsics = self.stream_profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics

            # Get depth scale
            self.depth_scale = self.depth_sensor.get_depth_scale()

            self.get_depth_units()


    def get_depth_units(self):
        # self.depth_sensor.set_option(rs.option.depth_units, du)
        self.depth_units = self.depth_sensor.get_option(rs.option.depth_units)
        self.depth_scale = self.depth_sensor.get_depth_scale()
        print(f"Depth units: {self.depth_units} - depth_scale: {self.depth_scale}")

    def next_frames(self):
        # Get the frame from the camera    
        frames = {}
        frame = self.queue.wait_for_frame()
        frameset = frame.as_frameset()
        if self.use_color_depth: frameset = self.align.process(frameset)
        if self.use_depth: 
            self.depth_frame = frameset.get_depth_frame()
            self.depth = np.asanyarray(self.depth_frame.get_data())
            if self.show_depth: 
                frames["color_depth"] = self.get_color_depth() 

        if self.use_color: 
            self.color_frame = frameset.get_color_frame()
            frames["color"] = np.asanyarray(self.color_frame.get_data()) 
            # fr_col = np.asanyarray(self.color_frame.get_data()) 
            # fr_bw = cv2.cvtColor(fr_col, cv2.COLOR_RGB2GRAY)
            # frames["color"] = cv2.cvtColor(fr_bw, cv2.COLOR_GRAY2RGB)
        return frames

    def get_color_depth(self):
        color_depth = np.asanyarray(self.colorizer.colorize(self.depth_frame).get_data())
        color_depth = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)
        return color_depth

    def deproject(self, x, y, d=None, averaging=False, zone_size=10):
        '''
        Find point in 3D camera coordinates (X,Y,Z) in m from 2D coordinates (x,y) in depth frame
        and return X,Y,Z
        x, y: coordinates in depth_frame
        d: distance in m. If None, depth is taken from current depth frame
        averaging : if False, the depth is taken on the one point (x,y).
                          if True, depth is averaged on a square zone of the current depth frame,
                          centered on (x, y) and of size 2*zone_size
        zone_size : size in pixel of the zone on which depth is averaged
        '''
        if d is None:
            if averaging:
                zone = self.depth[max(0,y-zone_size):min(self.res_h,y+zone_size),max(0,x-zone_size):min(self.res_w,x+zone_size)]
                zone = zone[zone!=0]
                d = 0 if zone.size == 0 else np.mean(zone[zone!=0]) * self.depth_scale
            else:
                d = self.depth[y,x] * self.depth_scale
        point = rs.rs2_deproject_pixel_to_point(self.intrinsics, [x, y], d)
        return point

    def project(self, Pc):
        '''
        Project a 3D camera coordinates point Pc [X Y Z] into the picture frame coordinates pixel (x,y)
        '''
        return tuple([ int(x) for x in rs.rs2_project_point_to_pixel(self.intrinsics, list(Pc)) ])

    def stop(self):
        self.pipe.stop()
    

if __name__ == "__main__":
    use_color = True
    use_depth = True
    res_w = 640
    res_h = 480

    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            X,Y,Z = s.deproject(x,y)
            print(f"X={X:.2f} Y={Y:.2f} Z={Z:.2f}")
    if use_color and use_depth:
        cv2.namedWindow("Color")
        cv2.setMouseCallback("Color", callback)

    s = RealsenseSensor(preset=1, res_w=res_w, res_h=res_h, use_depth=use_depth, use_color = use_color)
    while True:
        frames = s.next_frames()
        if use_color: cv2.imshow("Color", frames['color'])
        if use_depth: cv2.imshow("Depth", frames['color_depth'])
        k = cv2.waitKey(1)
        if k == 27:
            break
    s.stop()
    cv2.destroyAllWindows()