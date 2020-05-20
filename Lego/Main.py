
from robolink import *    # API to communicate with RoboDK
from robodk import *      # robodk robotics toolbox
import cv2 as cv
import numpy as np
#Our own scripts
import BrickDetection
import Calibration
import math


# Any interaction with RoboDK must be done through RDK: So those have to be written in order to communicate with RoboDk
RDK = Robolink()

#Closes any preopened camera view
RDK.Cam2D_Close() 

# Get the robot by name:
robot = RDK.Item('KUKA KR 6 R900 sixx', ITEM_TYPE_ROBOT)
tool = RDK.Item('Gripper', ITEM_TYPE_TOOL)

# Here we set the names of the obejcts in robodk
Calibration_bricks = ['YellowCalib','RedCalib','BlueCalib','GreenCalib']
Real_bricks= ['Blue 1','Blue 2','Green 1','Green 2','Red 1','Yellow 1','Yellow 2','Yellow 3','Yellow 4']


# This functions retrieves the location of the bricks used in the calibration
# This corresponds to putting the bricks on a known location in real life.
def getLocation(names):
    bricks = []
    for n in names:
        bricks.append(RDK.Item(n,ITEM_TYPE_OBJECT).Pose())
    return bricks

#This function is used to hide bricks when switching between calibration and the scenario
def hide_bricks(names):
    for n in names:
        RDK.Item(n,ITEM_TYPE_OBJECT).setVisible(False, visible_frame=None)
    return
#This function is used to show bricks when switching between calibration and the scenario
def show_bricks(names):
    for n in names:
        RDK.Item(n,ITEM_TYPE_OBJECT).setVisible(True, visible_frame=None)
    return



''' temp_target= RDK.Item('Temp target', ITEM_TYPE_TARGET)
frame = RDK.ItemUserPick('Select a reference frame', ITEM_TYPE_FRAME) 
brick1.setPoseFrame(frame) '''
# get the targets references by name:
home = RDK.Item('Home')
target = RDK.Item('Plane')

# get the pose of the target (4x4 matrix representing brick and orientation):
# This gives the brick of the target reference but as a mat type, but it's currently not used anywhere
#poseref = target.Pose() 
robot.setJoints([0,-90,90,180,-90,45])

#Select the camera and settings
#cameraf=RDK.ItemUserPick('Select the Camera location (reference, tool or object)')
cameraf = RDK.Item('CameraRef', ITEM_TYPE_FRAME)
cam_id = RDK.Cam2D_Add(cameraf, 'FOCAL_LENGHT=6 FOV=38 FAR_LENGHT=2100 SIZE=640x480 BG_COLOR=black LIGHT_SPECULAR=white')
#Take a snapshot of the cameraRef view then save it

#Init (Connect,setup) -> Place 4 bricks -> Take images -> find brick location -> Calibrate() -> Place coloured brickcs -> take images -> find brick location and orientation -> 
def captureImage(cam_id):
    path= RDK.getParam('PATH_OPENSTATION')
    RDK.Cam2D_Snapshot(path + "/image.png", cam_id)
    image=cv.imread(path + "/image.png")
    return image

#calibration part
#----------------------------------------------------------------------
# 4 bricks needs to be placed into the environment and the location is needed
# This is done automatically here:
show_bricks(Calibration_bricks)
hide_bricks(Real_bricks)
brick_loc_workspace = getLocation(Calibration_bricks)

#Here an image is capture with the 4 bricks on
image = captureImage(cam_id)

#cv.imshow("calibration bricks", image)
#cv.waitKey()

#A blob detection is then used to find the location of the bricks in the image
found_bricks = BrickDetection.GetBricks(image,verbose=False)

#Do the calibatrion to find d that is later used to map between the image coordinates and workspace
d = Calibration.calibrate(found_bricks,brick_loc_workspace)


#Computer Vision part
#----------------------------------------------------------------------
# The calibration bricks is hidden and the bricks to assembly is shown   
hide_bricks(Calibration_bricks)
show_bricks(Real_bricks)

# A image of the bricks is captured
image = captureImage(cam_id)

#cv.imshow("Bricks to pick up", image)
#cv.waitKey()

#The location and orientation of the bricks is found from the image
found_bricks = BrickDetection.GetBricks(image,verbose=False)


#Assembly List (Params: Brick_list)
#----------------------------------------------------------------------
    #Run through assembly order and check brick list if it has the brick color
    #Returns: Order of bricks 
# Mapping between image coordinates and workspace by use of d
def mapping(x,y,d):
    xC = ((d[0]*x)+(d[1]*y)+d[2])/((d[6]*x)+(d[7]*y)+1)
    yC = ((d[3]*x)+(d[4]*y)+d[5])/((d[6]*x)+(d[7]*y)+1)
    return xC,yC

# Use mapping to convert all brick locations into workspace   
def cameraToWorkspace(found_bricks,d):
    for brick in found_bricks:
        brick.pickup_position[0],brick.pickup_position[1] = mapping(brick.pickup_position[0],brick.pickup_position[1],d)
    return found_bricks

found_bricks = cameraToWorkspace(found_bricks,d)

#Used to make the order of how the bricks should be picked up
def GetOrder(blueprint,found_bricks):
    order = []
    for name in blueprint:
        for brick in found_bricks:
            if name == brick.color:
                order.append(brick)
                found_bricks.remove(brick)
                break
    return order

#Defining the colors needed to make the figures
bart = ["Yellow","Red","Blue","Yellow"]
marge = ["Yellow","Green","Green","Yellow","Blue"]


bartOrder = GetOrder(bart,found_bricks)
margeOrder = GetOrder(marge,found_bricks)



frame = RDK.Item('WorkSpace', ITEM_TYPE_FRAME)
robot.setPoseFrame(frame)
pose_ref = robot.Pose()

RDK.Cam2D_Close()

#This is where the robot is set to move
#It moves above each brick and picks it up
#Then it moves to the target location and places the brick down again
def Assemble(pose_ref,order,at_position):
    for i, brick in enumerate(order):
        #Rotation and move above pickup location
        pose_ref = TxyzRxyz_2_Pose([brick.pickup_position[0],brick.pickup_position[1],280,math.radians(-180),math.radians(0),math.radians(brick.slope)])
        robot.MoveJ(pose_ref)   
        
        #Pickup Object
        pose_ref.setPos(brick.pickup_position)
        robot.MoveL(pose_ref)
        tool.AttachClosest()

        #Home
        pose_ref.setPos([brick.pickup_position[0],brick.pickup_position[1],280])
        robot.MoveJ(pose_ref)

        #Above target and Rotation
        pose_ref = TxyzRxyz_2_Pose([at_position[0],at_position[1],300,math.radians(-180),math.radians(0),0])
        robot.MoveJ(pose_ref)   
        
        #Place Object
        pose_ref.setPos([at_position[0],at_position[1],25*i])
        robot.MoveL(pose_ref)
        tool.DetachAll()
        
        #Above target
        pose_ref.setPos([at_position[0],at_position[1],300])
        robot.MoveL(pose_ref)

Assemble(pose_ref, bartOrder,[240,430])
Assemble(pose_ref, margeOrder,[120,430])


if cv.waitKey():
    RDK.Cam2D_Close()
