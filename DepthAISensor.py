import cv2                                
import numpy as np   
import depthai
from time import time
from math import tan, pi, atan

# FOV for BW1092
HFOV_color = 65.9
VFOV_color = 40.06
HFOV_mono = 71.86
VFOV_mono = 44.3547

class DepthAISensor:
    def __init__(self, device_args=['', False], calib_file='calib.npz'):
        # Load retroprojection matrix Q from calib file
        npz = np.load(calib_file)
        self.Q = npz['Q']
        self.Qinv = np.linalg.pinv(self.Q)
        # DepthAI init
        self.device = depthai.Device(*device_args)
        config = {
            'streams': ['color', 'rectified_right', 'disparity'],
            # Even if we don't use NN here, still need to configure it
            'ai': {
                'blob_file': '/home/gx/oakd/depthai/resources/nn/landmarks-regression-retail-0009/landmarks-regression-retail-0009.blob.sh1cmx1NCE1',
                'camera_input': 'rgb', # or 'left', 'right'
                'shaves': 1,
                'cmx_slices': 1,
                'NN_engines': 1,
            },
            'depth': {
                'warp_rectify': {
                    'edge_fill_color': 0, # gray 0..255, or -1 (default) to replicate pixel values
                    # 'mirror_frame':
                    #    True (default): rectified is mirrored, disparity/depth is normal and mapped on `rectified_right` 
                    #    False         : rectified is normal, disparity/depth is mirrored and mapped on `rectified_left`
                    #'mirror_frame': False,
                },
            },
            'app': {
                'sync_sequence_numbers': True, # Default: False
                'usb_chunk_KiB': 0, # Default: 64. Larger than 64 improves throughput. 0 is best if it works as there's no chunking.
            },
        }
        self.pipeline = self.device.create_pipeline(config=config)
        if self.pipeline is None:
            raise RuntimeError("Error initializing pipeline")

        self.frames = {}
        self.frame_num = {}
        self.f_color = None # focal length of color camera
        self.f_mono = None # focal length of mono camera

    def next_frames(self):
        # Read frames until we get 3 frames (color, rectified_right, disparity) synchronized (same sequence number)
        found = False
        while not found:
            data_packets = self.pipeline.get_available_data_packets()
            for packet in data_packets:
                frame = packet.getData()
                
                if frame is None:
                    print('Invalid packet data!')
                    continue
                meta = packet.getMetadata()
                # print(packet.stream_name, meta.getSequenceNum())
                self.frame_num[packet.stream_name] = meta.getSequenceNum()
                if packet.stream_name == 'color':
                    if self.f_color is None:
                        self.w_color = meta.getFrameWidth()
                        self.h_color = meta.getFrameHeight()
                        self.f_color = self.w_color / (2 * tan(np.radians(HFOV_color/2)))
                    yuv420p = frame.reshape((self.h_color * 3 // 2, self.w_color))
                    self.frames['color'] = cv2.cvtColor(yuv420p, cv2.COLOR_YUV2BGR_IYUV)
                elif packet.stream_name == 'rectified_right':
                    if self.f_mono is None:
                        self.w_mono = meta.getFrameWidth()
                        self.h_mono = meta.getFrameHeight()
                        self.f_mono = self.w_mono / (2 * tan(np.radians(HFOV_mono/2)))
                    # Un-mirror the rectified streams
                    self.frames['rectified_right'] = cv2.flip(frame, 1)
                elif packet.stream_name == 'disparity':
                    self.frames['disparity'] = frame

                if self.frame_num.get('color', None) == self.frame_num.get('rectified_right', None) and \
                    self.frame_num.get('color', None) == self.frame_num.get('disparity', None) :
                    found = True
                    break
        return self.frames

    def deproject(self, x, y, averaging=False, zone_size=10):
        '''
        Find point in 3D camera coordinates (X,Y,Z) in m from 2D coordinates (x,y) in color frame
        and return X,Y,Z
        x, y: coordinates in color frame
        averaging : if False, the disparity is taken on the one point (x,y).
                          if True, depth is averaged on a square zone of the current depth frame,
                          centered on (x, y) and of size 2*zone_size
        zone_size : size in pixel of the zone on which disparity is averaged
        '''
        # Temporary solution until deproject available in depthai:
        # With BW1092, color camera closed to right camera
        # Map (x,y) from color frame to (xr,yr) in rectified_right frame
        xr = int(((x - self.w_color*0.5) * self.f_mono/ self.f_color + self.w_mono*0.5) )
        yr = int(((y - self.h_color*0.5) * self.f_mono/ self.f_color + self.h_mono*0.5) )
        if averaging:
            zone = self.frames['disparity'][max(0,yr-zone_size):min(self.h_mono,yr+zone_size),max(0,xr-zone_size):min(self.w_mono,xr+zone_size)]
            zone = zone[zone!=0]
            d = 0 if zone.size == 0 else np.mean(zone[zone!=0])
        else:
            d = self.frames['disparity'][yr,xr] 
        point = cv2.perspectiveTransform(np.array([[[xr,yr,d]]], dtype='float32'), self.Q)
        point = np.squeeze(point)/100
        return list(point)

    def project(self, P):
        '''
        Project a 3D camera coordinates point P [X Y Z] into the color frame coordinates pixel (x,y)
        '''
        pixel = cv2.perspectiveTransform(np.array([[P]], dtype='float32')*100, self.Qinv)        
        x, y, d= np.squeeze(pixel)
        # Map pixel (x,y) from rectified_right frame to (xc,yc) in color frame
        xc = int(((x - self.w_mono*0.5) * self.f_color/ self.f_mono + self.w_color*0.5) )
        yc = int(((y - self.h_mono*0.5) * self.f_color/ self.f_mono + self.h_color*0.5) )
        return (xc, yc)

    def stop(self):
        del self.pipeline
        del self.device





