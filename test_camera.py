import cameraModules.realsense.camera_rs as cam
import cv2 as cv
import numpy as np

rs = cam.RSCamera([640,480],30)

while True:
	# cv.imshow("Depth Image", depth)
	depth, color = rs.get_frames()
	cv.imshow("Color Image", color)
	cv.waitKey(1)
 
cv.destroyAllWindows()
