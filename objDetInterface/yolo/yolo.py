DEBUG = False
import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import torch
import cv2 as cv

# model = torch.hub.load(r'E:\University\RAIS\scripts\yolo_img_p\yolov5','custom',path=r"E:\University\RAIS\scripts\yolo_img_p\yolov5\runs\train\exp\weights\last.pt",source='local', force_reload=True)

class Yolo:
    def __init__(self, folder_path, model_path, conf):
        self.model = torch.hub.load(folder_path,'custom',path=model_path,source='local', force_reload=True)
        self.model.conf = conf
        
    
    def get_bound_box(self, frame):
        self.results = self.model(frame)
        self.df = self.results.pandas().xyxy[0]
        self.xmin = self.df.iloc[:,0]
        self.xmax = self.df.iloc[:,2]
        self.ymin = self.df.iloc[:,1]
        self.ymax = self.df.iloc[:,3]
         
        for i, pixel in self.xmin:
            pass
        
        
    def get_center_coord(self,frame):
        self.dot_coordinates = []
        self.results = self.model(frame)
        self.df = self.results.pandas().xyxy[0]
        self.xmin = self.df.iloc[:,0]
        self.xmax = self.df.iloc[:,2]
        self.ymin = self.df.iloc[:,1]
        self.ymax = self.df.iloc[:,3]

        self.image = frame.copy()
        
        for i in range(len(self.xmax)):
            self.bottomright = [int(self.xmax[i]),int(self.ymin[i])]
            self.topleft = [int(self.xmin[i]),int(self.ymax[i])]
            
            self.cX = int((self.topleft[0] + self.bottomright[0]) / 2.0)
            self.cY = int((self.topleft[1] + self.bottomright[1]) / 2.0)
        
            self.center = (self.cX,self.cY,0)#0 is the orientation, needs to be replaced with actual orientation value
            
            if DEBUG: 
                self.image = cv.circle(frame, (self.cX,self.cY), radius=0, color=(0, 0, 255), thickness=5) 
   
            self.dot_coordinates.append(self.center)
   
        if DEBUG:
            cv.imshow("obj det", self.image)
            cv.waitKey(0)
            cv.destroyAllWindows()
   
   
        return self.dot_coordinates
    
    def get_orientation(self,frame):
        temp=[]
        for i in range(len(self.dot_coordinates)):
            temp.append(0)
        return temp
    
    

if DEBUG:
    print("yolo module loaded successfully")