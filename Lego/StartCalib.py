from robolink import *    # API to communicate with RoboDK
from robodk import *      # robodk robotics toolbox
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import argparse
from numpy import loadtxt
from numpy import genfromtxt
from numpy import savetxt

# Any interaction with RoboDK must be done through RDK: So those have to be written in order to communicate with RoboDk
RDK = Robolink()
RDK.Cam2D_Close() #Closes any preopened camera view

# get the robot by name:
robot = RDK.Item('KUKA KR 6 R900 sixx', ITEM_TYPE_ROBOT)

# get the targets references by name:
home = RDK.Item('Home')
target = RDK.Item('Plane')

# get the pose of the target (4x4 matrix representing position and orientation):
poseref = target.Pose() # This gives the position of the target reference but as a mat type, but it's currently not used anywhere

# move the robot to home, then to the the : Uncheck to try
#robot.MoveJ(home)
#robot.MoveJ(poseref)
#print(robot.ConnectedState())
#"{} and {}".format("Robot connection:", robot.ConnectedState())


#get the size of the lego bricks from the user
PARAM_SIZE_BOX = 'SizeBox'
SIZE_BOX = RDK.getParam('SizeBox')
size_box = RDK.getParam(PARAM_SIZE_BOX)

size_box_input = mbox('Enter the size of the box in mm [L,W,H]', entry=size_box)
if size_box_input:
    RDK.setParam(PARAM_SIZE_BOX, size_box_input)
else:
    raise Exception('Operation cancelled by user')

#print(size_box_input)
robot.setJoints([90,-90,90,0,0,0])
#------------------------------------------------------------
#Select the camera and settings
cameraf=RDK.ItemUserPick('Select the Camera location (reference, tool or object)')
cam_id = RDK.Cam2D_Add(cameraf, 'FOCAL_LENGHT=6 FOV=38 FAR_LENGHT=2100 SIZE=640x480 BG_COLOR=black LIGHT_SPECULAR=white')

#-----------------------------------------------------------------------------------
#Get the height or whatever of the brick
SIZE_BOX_XYZ = [float(x.replace(' ','')) for x in SIZE_BOX.split(',')]
SIZE_BOX_Z=SIZE_BOX_XYZ[2]
#--------------------------------------------------------------------------
#Take a snapshot of the cameraRef view then save it
path= RDK.getParam('PATH_OPENSTATION')
RDK.Cam2D_Snapshot(path + "/image.png", cam_id)
#----------------------------------------------------------------------
#This is to locate the brick in the image, so change the location
#       to where you save your image in the previous step
imagepath=r'E:\OneDrive\OneDrive - Aalborg Universitet\AAU\8th semester\P8\DGCNN\SemesterProject_843\Lego\image.png'
image1=cv.imread(imagepath)
lower_color_bounds = np.array([0,100,0]) 
upper_color_bounds = np.array([0,255,80])

img_gray = cv.cvtColor(image1, cv.COLOR_BGR2GRAY) 
mask = cv.inRange(image1,lower_color_bounds,upper_color_bounds )
mask_rgb = cv.cvtColor(mask, cv.COLOR_GRAY2BGR) 
image = image1 & mask_rgb
im= cv.imread(imagepath,cv.IMREAD_GRAYSCALE)
detector=cv.SimpleBlobDetector_create()
keypoints = detector.detect(im)
im_with_keypoints=cv.drawKeypoints(im,keypoints,np.array([]),(0,0,255),cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

brick_loc=[]
for i in keypoints:
    x = i.pt[0] #i is the index of the blob you want to get the position
    y = i.pt[1]
    brick_loc.append([[x],[y],[1]])

cv.imshow("whatever",im_with_keypoints) 

edges = cv.Canny(img_gray,50,150,apertureSize = 3)
cv.imshow("whatever2",edges)
lines = cv.HoughLines(edges,1,np.pi/180,200)
for rho,theta in lines[0]:
    ab = np.cos(theta)
    bb = np.sin(theta)
    x0 = ab*rho
    y0 = bb*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv.imwrite('houghlines3.jpg',image1)

#mtx is the cameras intrinsic and extrinsic parameters
mtx = genfromtxt('E:\OneDrive\OneDrive - Aalborg Universitet\AAU\8th semester\RobotVision\Miniproject\Lego\mtx.csv', delimiter=',')
rvecs = genfromtxt('E:\OneDrive\OneDrive - Aalborg Universitet\AAU\8th semester\RobotVision\Miniproject\Lego\ecs.csv', delimiter=',')
print(brick_loc)

""" 
x1 = brick_loc[3][0][0]
y1 = brick_loc[3][1][0]
x1c = 0.0
y1c = 0.0

x2 = brick_loc[2][0][0]
y2 = brick_loc[2][1][0]
x2c = 1000.0
y2c = 0.0

x3 = brick_loc[1][0][0]
y3 = brick_loc[1][1][0]
x3c = 0.0
y3c = 1000.0

x4 = brick_loc[0][0][0]
y4 = brick_loc[0][1][0]
x4c = 1000
y4c = 1000

a = np.array(
    [[x1,0,x2,0,x3,0,x4,0],
    [y1,0,y2,0,y3,0,y4,0],
    [1,0,1,0,1,0,1,0],
    [0,x1,0,x2,0,x3,0,x4],
    [0,y1,0,y2,0,y3,0,y4],
    [0,1,0,1,0,1,0,1],
    [-x1*x1c,-x1*y1c,-x2*x2c,-x2*y2c,-x3*x3c,-x3*y3c,-x4*x4c,-x4*y4c],
    [-y1*x1c,-y1*y1c,-y2*x2c,-y2*y2c,-y3*x3c,-y3*y3c,-y4*x4c,-y4*x4c]])

a = np.transpose(a)
 """
#b = [x1c,y1c,x2c,y2c,x3c,y3c,x4c,y4c]

""" def mapping(x,y,d):
    xC = ((d[0]*x)+(d[1]*y)+d[2])/((d[6]*x)+(d[7]*y)+1)
    yC = ((d[3]*x)+(d[4]*y)+d[5])/((d[6]*x)+(d[7]*y)+1)
    return xC,yC

d=np.linalg.solve(a,b) """
#savetxt(r'E:\OneDrive\OneDrive - Aalborg Universitet\AAU\8th semester\P8\DGCNN\SemesterProject_843\Lego\d.csv',d, delimiter='')

""" testX,testY= mapping(x4,y4,d)

print(testX,testY)
 """


#print(np.allclose(np.dot(a, x), b))



if cv.waitKey():
    RDK.Cam2D_Close() #Closes any preopened camera view