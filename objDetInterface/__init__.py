DEBUG = True

import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)


class ObjDet:
    if DEBUG:
        print("created object")
        
    def __init__(self, type, folder_path=None, model_path=None, bg=None, conf = None):
        if type == 'yolo':
            import yolo.yolo as od
            self.obj_det = od.Yolo(folder_path, model_path, conf)
            
        elif type == 'BG':
            import BGSub.BGSub as od
            self.obj_det = od.BGSub(bg)
            
            
    def get_center_coord(self,frame):
        return self.obj_det.get_center_coord(frame)
    
    
    def get_orientation(self,frame):
        self.obj_det.get_orientation(frame)
        
    def update_bg(self,bg):
        self.obj_det.update_bg(bg)