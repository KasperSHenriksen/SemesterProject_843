from robolink import *    # API to communicate with RoboDK
from robodk import *      # robodk robotics toolbox
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import argparse
from numpy import loadtxt
from numpy import genfromtxt
from numpy import savetxt

#class for the bricks that contains all nesesary information
class Brick:
    def __init__(self):
        self.color = 'None' #Which color it is
        self.pickup_position = [0,0,0] #Pickup position
        self.slope = 0 #The rotation of it
    def summary(self):
        print(f'Color: {self.color}, position: {self.pickup_position}, slope: {self.slope}')


#COLOR DETECTION --> Thresholds: HSV
def DetectColor(image,lower_boundary,upper_boundary):
    """
    Creates a colored image within a given image based on specified boundaries. 

    Parameters:
    ----------
    image : mat
        HSV screenshot from robot camera 
    lower_boundary : np.array 
        Minimum color threshold boundary based on [H,S,V]
    upper_boundary : np.array 
        Maximum color threshold boundary based on [H,S,V]

    Returns:
    -------
    mat
        A colored image, containing the colors within the given threshold
    """
    mask = cv.inRange(image, lower_boundary, upper_boundary) #Thresholding
    color_image = cv.bitwise_and(image,image, mask= mask) #Bitwise AND operation, which returns 1 for each pixel if both images contain 1 in a pixel. 
    return color_image

#-------------------------------------------------------
#Find the location for each point
def FindBricks(detector, hsv_color_image, color_name):
    """
    Creates a colored image within a given image based on specified boundaries. 

    Parameters:
    ----------
    detecor : SimpleBlobDetector
        SimpleBlobDetector from OpenCV 
    hsv_color_image : mat
        HSV based image [H,S,V]
    color_name : String
        Name of the colored blobs that should be detected

    Returns:
    -------
    list<Brick>
        Containing entities of the Brick Class
    """
    _,_,v= cv.split(hsv_color_image) #Splitting channels of HSV image into H S V
    grayscale_img = cv.bitwise_not(v) #Bitwise NOT operation, returns 0 for each pixel, if image contains 1 and opposite.
    keypoints = detector.detect(grayscale_img) #Detects the keypoint structs, which are the blobs containing e.g. position.
    grayscale_img = cv.bitwise_not(grayscale_img) #Bitwise NOT operation, returns 0 for each pixel, if image contains 1 and opposite.

    found_bricks = [] #Empty list, stores entities of Brick Class
    for keypoint in keypoints: #Goes through each keypoint
        brick = Brick() #Initialize a new entitity of Brick Class
        brick.color = color_name #Sets it a name based on its color
        x = keypoint.pt[0] #X coordinate of blob
        y = keypoint.pt[1] #Y coordinate of blob
        brick.pickup_position = [y,x,0] #Sets the desired pickup position.
        found_bricks.append(brick) #Appends the Brick instance
    return found_bricks


#-------------------------------------------------------
#Extract the patches
def ExtractPatches(brick_location,colored_image):
    """
    Extracts a patch of where the brick is found. This is used for independently compute Hough Lines for one blob at a time.

    Parameters:
    ----------
    brick_location : list<float>
        The position of the brick, [x,y,z]
    colored_image : mat
        HSV based image [H,S,V]

    Returns:
    -------
    mat
        A cropped image containing the Brick only and some padding

    """

    patch_center = np.array([brick_location[0], brick_location[1]]) #The center location of the patch
    patch_scale = 0.18 #The area to be patched

    smaller_dim = np.min(colored_image.shape[0:2]) #
    patch_size = int(patch_scale * smaller_dim)
    patch_x = int(patch_center[0] - patch_size / 2.)
    patch_y = int(patch_center[1] - patch_size / 2.)

    patch_image = colored_image[patch_x:patch_x+patch_size, patch_y:patch_y+patch_size] #Cropping the colored image based on patch dimensions
    patch_image = cv.morphologyEx(patch_image, cv.MORPH_CLOSE, np.ones((5,5))) #Morphology CLOSE operation using kernel 5x5 to close holes in blob.
    return patch_image

def ResizePatch(patch_image):
    """
    Enlarges a given patch.

    Parameters:
    ----------
    patch_image : mat
        A cropped image, which is a patch containing a Brick and some padding.

    Returns:
    -------
    mat
        Enlarged patch of a Brick
    """

    scale_percent = 220 #Percent of original size
    width = int(patch_image.shape[1] * scale_percent / 100) #
    height = int(patch_image.shape[0] * scale_percent / 100)
    dim = (width, height) #Dimensions of new patch

    resized_patch_image = cv.resize(patch_image, dim, interpolation = cv.INTER_AREA) #Resized the patch to the specified dimension. Moreover, it used the interpolation method INTER_AREA.
    return resized_patch_image

def ComputeSlope(image,image2):
    """
    To achieve the rotation of a Brick, Canny based edge detection and Hough Lines are used.

    Parameters:
    ----------
    image : mat
        A enlarged patch of a Brick
    image2 : mat
        The original image based on HSV

    Returns:
    -------
    float
        Slope of a Brick
    """

    closing = cv.morphologyEx(image, cv.MORPH_CLOSE, np.ones((5,5))) #Morphology CLOSE operation.
    closing = cv.Canny(closing, 50, 200, None, 3) #Canny edge detection
    closing = cv.dilate(closing,(3,3),iterations = 1) #Morpology DILATE operation. 
    lines = cv.HoughLinesP(closing,1,np.pi/180,10,10,10) #Calculate Hough Lines to find the rotation of the brick.
    if lines is not None: 
        for line in lines: #Go through each found line based on Hough Lines
            for x1,y1,x2,y2 in line:
                cv.line(image,(x1,y1),(x2,y2),(0,0,255),2)       
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
    hsv_img = cv.cvtColor(image1, cv.COLOR_BGR2HSV) #Converts BGR Image to HSV Image
    red_hsv_color = DetectColor(hsv_img,lower_boundary = np.array([-15,90,220]), upper_boundary = np.array([15,255,255])) #Red Detected 
    green_hsv_color = DetectColor(hsv_img,lower_boundary = np.array([138/2-15,90,220]), upper_boundary = np.array([138/2+15,255,255])) #Green Detected
    yellow_hsv_color = DetectColor(hsv_img,lower_boundary = np.array([57/2-15,90,220]), upper_boundary = np.array([59/2+15,255,255])) #Yellow Detected
    blue_hsv_color = DetectColor(hsv_img,lower_boundary = np.array([239/2-15,90,220]), upper_boundary = np.array([239/2+15,255,255])) #Blue Detected
    detected_color_dict = {"Red":red_hsv_color, "Green":green_hsv_color, "Yellow":yellow_hsv_color, "Blue":blue_hsv_color} #Dictionary of colored images


    #Blob Detection & Get Bricks
    detector=cv.SimpleBlobDetector_create() #Instance of SimpleBlobDetector
    brick_list = [] #Empty list, used for appending bricks.
    for color_name in detected_color_dict: #Goes through one detected color at a time.
        found_bricks = FindBricks(detector, hsv_color_image = detected_color_dict.get(color_name), color_name = color_name) #Find Bricks and return them.
        brick_list.extend(found_bricks) #Extend list of Bricks

    #Calculate Slope using Haugh Lines
    for brick in brick_list: #Goes through each found brick
        detected_color_image = detected_color_dict.get(brick.color) #Gets the colored image that correlates to the color of the Brick.
        extracted_patch = ExtractPatches(brick.pickup_position, detected_color_image) #Extracts a patch of where the Brick is.
        resized_patch = ResizePatch(extracted_patch) #Enlargens the patch to easier be used to calculate its slope
        slope = ComputeSlope(resized_patch, image1) #Computes the slope based on Canny Detection and Hough Lines
        brick.slope = slope #Sets the slope of the brick
    
    #Summary, Show information of the detected bricks
    if verbose is True:
        for brick in brick_list:
            brick.summary()

    return brick_list