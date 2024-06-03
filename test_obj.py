import objDetInterface as objDet
import cv2 as cv
import cameraModules.realsense.camera_rs as cam
import numpy as np

rs = cam.RSCamera([640,480],30)
_, color = rs.get_frames()
_, color = rs.get_frames()

obj = objDet.ObjDet('yolo',folder_path=r'E:\University\RAIS\scripts\yolo_img_p\yolov5', model_path=r"E:\University\RAIS\scripts\yolo_img_p\yolov5\runs\train\exp\weights\last.pt", conf=0.92)
# obj = objDet.ObjDet('BG',bg='bg.jpg')


# imgg = color
# imgg = cv.imread('img6.jpg')
# imgg2 = cv.imread('bg.jpg')
# print(type(imgg2))
# cv.imshow('asdasd',imgg2)
# cv.waitKey()
center_coords = obj.get_center_coord(np.array(color))
# orientation = obj.get_orientation(imgg)
# print(orientation)
print(center_coords)