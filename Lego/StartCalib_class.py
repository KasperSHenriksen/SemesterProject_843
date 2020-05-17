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
imagepath=r'/home/kasper/Desktop/GitHub/SemesterProject_843/Lego/image.png'
image1=cv.imread(imagepath)

#cv.imshow('original_img',image1)
cv.imwrite(path+"/originalimage.jpg",image1)

#class
class Brick:
    def __init__(self):
        self.color = 'None'
        self.position = [0,0,0]
        self.slope = 0
    def summary(self):
        print(f'Color: {self.color}, position: {self.position}, slope: {self.slope}')


#COLOR DETECTION --> Thresholds: HSV
def DetectColor(image,lower_boundary,upper_boundary):
    mask = cv.inRange(image, lower_boundary, upper_boundary)
    color_image = cv.bitwise_and(image,image, mask= mask)
    return color_image

hsv_img = cv.cvtColor(image1, cv.COLOR_BGR2HSV)
red_hsv_color = DetectColor(hsv_img,lower_boundary = np.array([-15,90,220]), upper_boundary = np.array([15,255,255]))
green_hsv_color = DetectColor(hsv_img,lower_boundary = np.array([138/2-15,90,220]), upper_boundary = np.array([138/2+15,255,255]))
yellow_hsv_color = DetectColor(hsv_img,lower_boundary = np.array([57/2-15,90,220]), upper_boundary = np.array([59/2+15,255,255]))
detected_color_dict = {"Red":red_hsv_color,"Green":green_hsv_color,"Yellow":yellow_hsv_color}

#-------------------------------------------------------
#Find the location for each point
def FindBricks(hsv_color_image, color_name):
    _,_,v= cv.split(hsv_color_image)
    grayscale_img = cv.bitwise_not(v)
    keypoints = detector.detect(grayscale_img)
    grayscale_img = cv.bitwise_not(grayscale_img) #Why is this here? Debug later

    found_bricks = []
    for keypoint in keypoints:
        brick = Brick()
        brick.color = color_name
        x = keypoint.pt[0] #i is the index of the blob you want to get the position
        y = keypoint.pt[1]
        brick.position = [y,x,1]
        found_bricks.append(brick)
    return found_bricks

detector=cv.SimpleBlobDetector_create()

brick_list = []
for color_name in detected_color_dict:
    found_bricks = FindBricks(hsv_color_image = detected_color_dict.get(color_name), color_name = color_name)
    brick_list.extend(found_bricks)

#-------------------------------------------------------
#Extract the patches
def ExtractPatches(brick_location,colored_image):
    patch_center = np.array([brick_location[0], brick_location[1]]) # the location of the patch
    patch_scale = 0.18# the area to be patched

    smaller_dim = np.min(colored_image.shape[0:2])
    patch_size = int(patch_scale * smaller_dim)
    patch_x = int(patch_center[0] - patch_size / 2.)
    patch_y = int(patch_center[1] - patch_size / 2.)

    patch_image = colored_image[patch_x:patch_x+patch_size, patch_y:patch_y+patch_size]
    patch_image = cv.morphologyEx(patch_image, cv.MORPH_CLOSE, np.ones((5,5)))
    return patch_image

def ResizePatch(patch_image):
    scale_percent = 220 # percent of original size
    width = int(patch_image.shape[1] * scale_percent / 100)
    height = int(patch_image.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized_patch_image = cv.resize(patch_image, dim, interpolation = cv.INTER_AREA)
    return resized_patch_image

def ComputeSlope(image,image2):
    closing = cv.morphologyEx(image, cv.MORPH_CLOSE, np.ones((5,5)))
    closing = cv.Canny(closing, 50, 200, None, 3)
    closing = cv.dilate(closing,(3,3),iterations = 1)
    lines = cv.HoughLinesP(closing,1,np.pi/180,10,10,10)
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv.line(image2,(x1,y1),(x2,y2),(0,0,0),2)
                slope = (y2-y1)/(x2-x1)
                slope = math.atan(slope)
                slope = math.degrees(slope)
                return slope

for brick in brick_list:
    detected_color_image = detected_color_dict.get(brick.color)
    extracted_patch = ExtractPatches(brick.position, detected_color_image)
    resized_patch = ResizePatch(extracted_patch)
    slope = ComputeSlope(resized_patch, image1)
    brick.slope = slope


# for brick in brick_list:
#     brick.summary()

bart = ['Red', 'Yellow', 'Green']
for desired_brick in bart:
    print(desired_brick)





""" cv.imshow("Redimage",closing)
cv.imshow("Greenimage",GreenImage)
cv.imshow("Yellowimage",YellowImage)
 """

""" boundaries = [
	([0, 200, 0], [0, 255, 80]),
	([0, 230, 230], [0, 250, 255])
]
my_Images = []
for (lower, upper) in boundaries:
    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")
    mask = cv.inRange(image1, lower, upper)
    cv.imshow("mask",mask)
    output = cv.bitwise_and(image1, image1, mask = mask)
    #closing = cv.morphologyEx(output, cv.MORPH_CLOSE, np.ones((5,5)))
    my_Images.append(output)

cv.imshow("Green",my_Images[0])
cv.imshow("Blue",my_Images[1])
cv.imshow("Red",my_Images[2]) """




#mtx is the cameras intrinsic and extrinsic parameters
#mtx = genfromtxt('E:\OneDrive\OneDrive - Aalborg Universitet\AAU\8th semester\RobotVision\Miniproject\Lego\mtx.csv', delimiter=',')
#rvecs = genfromtxt('E:\OneDrive\OneDrive - Aalborg Universitet\AAU\8th semester\RobotVision\Miniproject\Lego\ecs.csv', delimiter=',')
#print(brick_loc)
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

b = [x1c,y1c,x2c,y2c,x3c,y3c,x4c,y4c]
 """
def mapping(x,y,d):
    xC = ((d[0]*x)+(d[1]*y)+d[2])/((d[6]*x)+(d[7]*y)+1)
    yC = ((d[3]*x)+(d[4]*y)+d[5])/((d[6]*x)+(d[7]*y)+1)
    return xC,yC

#d=np.linalg.solve(a,b)
#savetxt(r'E:\OneDrive\OneDrive - Aalborg Universitet\AAU\8th semester\P8\DGCNN\SemesterProject_843\Lego\d.csv',d, delimiter='')

""" testX,testY= mapping(x4,y4,d)

print(testX,testY) """

#print(np.allclose(np.dot(a, x), b))


cv.startWindowThread()

cv.waitKey(0)


# if cv.waitKey():
#     RDK.Cam2D_Close() #Closes any preopened camera view
