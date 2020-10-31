import numpy as np

# Double exponential filter
# Adaptation of https://social.msdn.microsoft.com/Forums/en-US/850b61ce-a1f4-4e05-a0c9-b0c208276bec/joint-smoothing-code-for-c?forum=kinectv2sdk
# smoothing: [0..1], lower values closer to raw data. Will lag when too high
# correction: [0..1], How much to correct back from prediction.  Can make things springy. 
#                     Lower values slower to correct towards the raw data
# prediction: [0..n], the number of frames to predict into the future. 
#                     Can over shoot when too high
# jitter_radius: Size in meter of the radius where jitter is removed. 
#                Can do too much smoothing when too high
#      Jitter radius is a clamp on frame-over-frame variations in the joint position. 
#      Before the double-exponential filter is applied, the raw (unfiltered) position 
#      is compared to the filtered position that was calculated for the previous frame. 
#      If this difference exceeds the jitter radius, the value will be clamped to 
#      the jitter radius before it goes into the double-exponential filter.
# max_deviation_radius: The maximum radius in meters that filtered positions are allowed to deviate from raw data
#                       Can snap back to noisy data when too high

class DoubleExpFilter:
    def __init__(self,smoothing=0.65,
                 correction=1.0,
                 prediction=0.85,
                 jitter_radius=250.,
                 max_deviation_radius=540.,
                 out_int=False):
        self.smoothing = smoothing
        self.correction = correction
        self.prediction = prediction
        self.jitter_radius = jitter_radius
        self.max_deviation_radius = max_deviation_radius
        self.count = 0
        self.filtered_pos = 0
        self.trend = 0
        self.raw_pos = 0
        self.out_int = out_int
        self.enable_scrollbars = False
    
    def reset(self):
        self.count = 0
        self.filtered_pos = 0
        self.trend = 0
        self.raw_pos = 0
    
    def update_config(self,smoothing,correction,prediction,jitter_radius,max_deviation_radius):
        self.smoothing = smoothing
        self.correction = correction
        self.prediction = prediction
        self.jitter_radius = jitter_radius
        self.max_deviation_radius = max_deviation_radius
        print("Smoothing params:",smoothing,correction,prediction,jitter_radius,max_deviation_radius)
        
    def update(self, pos):
        raw_pos = np.asanyarray(pos)
        if self.count > 0:
            prev_filtered_pos = self.filtered_pos
            prev_trend = self.trend
            prev_raw_pos = self.raw_pos
        
        if self.count == 0:
            self.shape = raw_pos.shape
            filtered_pos = raw_pos
            trend = np.zeros(self.shape)
            self.count = 1
        elif self.count == 1:
            filtered_pos = (raw_pos + prev_raw_pos)/2
            diff = filtered_pos - prev_filtered_pos
            trend = diff*self.correction + prev_trend*(1-self.correction)
            self.count = 2
        else:
            # First apply jitter filter
            diff = raw_pos - prev_filtered_pos
            length_diff = np.linalg.norm(diff)
            if length_diff <= self.jitter_radius:
                alpha = pow(length_diff/self.jitter_radius,1.5)
                # alpha = length_diff/self.jitter_radius
                filtered_pos = raw_pos*alpha \
                                + prev_filtered_pos*(1-alpha)
            else:
                filtered_pos = raw_pos
            
            # Now the double exponential smoothing filter
            filtered_pos = filtered_pos*(1-self.smoothing) \
                        + self.smoothing*(prev_filtered_pos+prev_trend)
            diff = filtered_pos - prev_filtered_pos
            trend = self.correction*diff + (1-self.correction)*prev_trend
        
        # Predict into the future to reduce the latency
        predicted_pos = filtered_pos + self.prediction*trend
        
        # Check that we are not too far away from raw data
        diff = predicted_pos - raw_pos
        length_diff = np.linalg.norm(diff)
        if length_diff > self.max_deviation_radius:
            predicted_pos = predicted_pos*self.max_deviation_radius/length_diff \
                        + raw_pos*(1-self.max_deviation_radius/length_diff)
        # Save the data for this frame
        self.raw_pos = raw_pos
        self.filtered_pos = filtered_pos
        self.trend = trend
        
        # Output the data
        #print("update a:",pos)
        #print("update b:", predicted_pos)
        if self.out_int:
            return predicted_pos.astype(int)
        else:
            return predicted_pos

    def scrollbars(self, enable=False, position=None):
        '''
        position = None or (x,y)
        '''
        import cv2
        if self.enable_scrollbars:
            if not enable:
                cv2.destroyWindow("DoubleExpFilter")
                self.enable_scrollbars = False
        else:
            if enable:

                cv2.namedWindow("DoubleExpFilter")
                if position: cv2.moveWindow("DoubleExpFilter", position[0],position[1])
                cv2.createTrackbar('smoothing','DoubleExpFilter', int(self.smoothing*100) ,100, lambda x: self.cb_smoothing(x, self))
                cv2.createTrackbar('correction','DoubleExpFilter', int(self.correction*100),100, lambda x: self.cb_correction(x, self))
                cv2.createTrackbar('prediction','DoubleExpFilter', int(self.prediction*10) ,100, lambda x: self.cb_prediction(x, self))
                cv2.createTrackbar('jitter','DoubleExpFilter', int(self.jitter_radius), 1000, lambda x: self.cb_jitter(x, self))
                cv2.createTrackbar('maxdev','DoubleExpFilter', int(self.max_deviation_radius),1000, lambda x: self.cb_maxdev(x, self))
                self.enable_scrollbars = True

    @staticmethod
    def cb_smoothing(value, self):
        self.smoothing = value/100
    @staticmethod
    def cb_correction(value, self):
        self.correction = value/100
    @staticmethod
    def cb_prediction(value, self):
        self.prediction = value/10
    @staticmethod
    def cb_jitter(value, self):
        self.jitter_radius = value
    @staticmethod
    def cb_maxdev(value, self):
        self.max_deviation_radius = value
                

class SimpleExpFilter:
    """
    return a * x + (1 - a) * x_prev
    """
    def __init__(self, alpha=0.65, out_int=False):
        self.alpha = alpha
        self.out_int = out_int
        self.count = 0
        self.init = True
        self.enable_scrollbars = False
           
    def reset(self):
        self.init = True

    def update(self, x):
        x = np.asanyarray(x)      
        if self.init:
            self.x_prev = x
            self.count = 1
            self.init = False
        else:
            self.x_prev = self.alpha * x + (1 - self.alpha) * self.x_prev
        if self.out_int:
            return self.x_prev.astype(int)
        else:
            return self.x_prev
            
    def scrollbars(self, enable=False, position=None):
        '''
        position = None or (x,y)
        '''
        import cv2
        if self.enable_scrollbars:
            if not enable:
                cv2.destroyWindow("SimpleExpFilter")
                self.enable_scrollbars = False
        else:
            if enable:
                cv2.namedWindow("SimpleExpFilter")
                if position: cv2.moveWindow("SimpleExpFilter", position[0],position[1])
                cv2.createTrackbar('alpha','SimpleExpFilter', int(self.alpha*100) ,100, lambda x: self.cb_alpha(x, self))
                self.enable_scrollbars = True

    @staticmethod
    def cb_alpha(value, self):
        self.alpha = value/100