from robolink import *    # API to communicate with RoboDK
from robodk import *      # robodk robotics toolbox
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import argparse
from numpy import loadtxt
from numpy import genfromtxt
from numpy import savetxt

    # # Any interaction with RoboDK must be done through RDK: So those have to be written in order to communicate with RoboDk
    # RDK = Robolink()
    # RDK.Cam2D_Close() #Closes any preopened camera view

    # # get the robot by name:
    # robot = RDK.Item('KUKA KR 6 R900 sixx', ITEM_TYPE_ROBOT)

    # # get the targets references by name:
    # home = RDK.Item('Home')
    # target = RDK.Item('Plane')

    # # get the pose of the target (4x4 matrix representing position and orientation):
    # poseref = target.Pose() # This gives the position of the target reference but as a mat type, but it's currently not used anywhere

# move the robot to home, then to the the : Uncheck to try
#robot.MoveJ(home)
#robot.MoveJ(poseref)
#print(robot.ConnectedState())
#"{} and {}".format("Robot connection:", robot.ConnectedState())


#get the size of the lego bricks from the user
# PARAM_SIZE_BOX = 'SizeBox'
# SIZE_BOX = RDK.getParam('SizeBox')
# size_box = RDK.getParam(PARAM_SIZE_BOX)

#size_box_input = mbox('Enter the size of the box in mm [L,W,H]', entry=size_box)
#size_box_input = (5,5,5)
# if size_box_input:
#     RDK.setParam(PARAM_SIZE_BOX, size_box_input)
# else:
#     raise Exception('Operation cancelled by user')

#print(size_box_input)
    # robot.setJoints([90,-90,90,0,0,0])
    # #------------------------------------------------------------
    # #Select the camera and settings
    # #cameraf=RDK.ItemUserPick('Select the Camera location (reference, tool or object)')
    # cameraf = RDK.Item('CameraRef', ITEM_TYPE_FRAME)
    # cam_id = RDK.Cam2D_Add(cameraf, 'FOCAL_LENGHT=6 FOV=38 FAR_LENGHT=2100 SIZE=640x480 BG_COLOR=black LIGHT_SPECULAR=white')

    # #-----------------------------------------------------------------------------------
    # #Get the height or whatever of the brick
    # # SIZE_BOX_XYZ = [float(x.replace(' ','')) for x in SIZE_BOX.split(',')]
    # # SIZE_BOX_Z=SIZE_BOX_XYZ[2]
    # #--------------------------------------------------------------------------
    # #Take a snapshot of the cameraRef view then save it
    # path= RDK.getParam('PATH_OPENSTATION')
    # RDK.Cam2D_Snapshot(path + "/image.png", cam_id)

    # print(path)
    # #----------------------------------------------------------------------
    # #This is to locate the brick in the image, so change the location
    # #       to where you save your image in the previous step
    # imagepath=r'E:\OneDrive\OneDrive - Aalborg Universitet\AAU\8th semester\P8\DGCNN\SemesterProject_843\Lego\image.png'
    # image1=cv.imread(imagepath)

    # #cv.imshow('original_img',image1)
    # cv.imwrite(path+"/originalimage.jpg",image1)

#class
class Brick:

    def __init__(self):
        self.color = 'None'
        self.pickup_position = [0,0,0] #Pickup and add place
        self.desired_position = [0,0,0]
        self.slope = 0
    def summary(self):
        print(f'Color: {self.color}, position: {self.pickup_position}, slope: {self.slope}')


#COLOR DETECTION --> Thresholds: HSV
def DetectColor(image,lower_boundary,upper_boundary):
    mask = cv.inRange(image, lower_boundary, upper_boundary)
    color_image = cv.bitwise_and(image,image, mask= mask)
    return color_image

#-------------------------------------------------------
#Find the location for each point
def FindBricks(detector, hsv_color_image, color_name):
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
        brick.pickup_position = [y,x,0]
        found_bricks.append(brick)
    return found_bricks


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




def GetBricks(image1,verbose):
    """
    Uses a captured image and runs Color Detection, Blob Detection, Compute Slope using haugh Lines respectively.
    
    Parameters
    ----------
    arg1 : mat
        Screenshot from robot camera
    
    Returns
    -------
    Brick Class
        Contains color, positions and slope
    """

    #Color Detection
    hsv_img = cv.cvtColor(image1, cv.COLOR_BGR2HSV)
    red_hsv_color = DetectColor(hsv_img,lower_boundary = np.array([-15,90,220]), upper_boundary = np.array([15,255,255]))
    green_hsv_color = DetectColor(hsv_img,lower_boundary = np.array([138/2-15,90,220]), upper_boundary = np.array([138/2+15,255,255]))
    yellow_hsv_color = DetectColor(hsv_img,lower_boundary = np.array([57/2-15,90,220]), upper_boundary = np.array([59/2+15,255,255]))
    blue_hsv_color = DetectColor(hsv_img,lower_boundary = np.array([239/2-15,90,220]), upper_boundary = np.array([239/2+15,255,255]))
    detected_color_dict = {"Red":red_hsv_color, "Green":green_hsv_color, "Yellow":yellow_hsv_color, "Blue":blue_hsv_color}


    #Blob Detection & Get Bricks
    detector=cv.SimpleBlobDetector_create()
    brick_list = []
    for color_name in detected_color_dict:
        found_bricks = FindBricks(detector, hsv_color_image = detected_color_dict.get(color_name), color_name = color_name)
        brick_list.extend(found_bricks)

    #Calculate Slope using Haugh Lines
    for brick in brick_list:
        detected_color_image = detected_color_dict.get(brick.color)
        extracted_patch = ExtractPatches(brick.pickup_position, detected_color_image)
        resized_patch = ResizePatch(extracted_patch)
        slope = ComputeSlope(resized_patch, image1)
        brick.slope = slope    
    
    #Summary, Show information of the detected bricks
    if verbose is True:
        for brick in brick_list:
            brick.summary()

    return brick_list