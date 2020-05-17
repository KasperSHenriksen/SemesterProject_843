
from robolink import *    # API to communicate with RoboDK
from robodk import *      # robodk robotics toolbox
import cv2 as cv
import numpy as np
#Our own scripts
import BrickDetection
import Calibration


# Any interaction with RoboDK must be done through RDK: So those have to be written in order to communicate with RoboDk
RDK = Robolink()

#Closes any preopened camera view
RDK.Cam2D_Close() 

# get the robot by name:
robot = RDK.Item('KUKA KR 6 R900 sixx', ITEM_TYPE_ROBOT)

Calibration_bricks = ['YellowCalib','RedCalib','BlueCalib','GreenCalib']
Real_bricks= ['Blue 1','Blue 2','Green 1','Green 2','Red 1','Yellow 1','Yellow 2','Yellow 3','Yellow 4']

def getLocation(names):
    bricks = []
    for n in names:
        bricks.append(RDK.Item(n,ITEM_TYPE_OBJECT).Pose())
    return bricks


def hide_bricks(names):
    for n in names:
        RDK.Item(n,ITEM_TYPE_OBJECT).setVisible(False, visible_frame=None)
    return

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

# get the pose of the target (4x4 matrix representing position and orientation):
# This gives the position of the target reference but as a mat type, but it's currently not used anywhere
poseref = target.Pose() 
robot.setJoints([0,-90,90,180,-90,0])

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

#calibration part, only needs to be run if the camera is moved.
#----------------------------------------------------------------------
# 4 bricks needs to be placed into the environment and the location of them needs to be pasted into brick_loc_workspace
show_bricks(Calibration_bricks)
brick_loc_workspace = getLocation(Calibration_bricks)
print("----------")
print(brick_loc_workspace[0].Cols()[3][0])
print("----------")

hide_bricks(Real_bricks)
image = captureImage(cam_id)
cv.imshow("cal", image)
found_bricks = BrickDetection.GetBricks(image,verbose=False)

#Do the calibatrion
d = Calibration.calibrate(found_bricks,brick_loc_workspace)


#Computer Vision part
#----------------------------------------------------------------------
hide_bricks(Calibration_bricks)
show_bricks(Real_bricks)
image = captureImage(cam_id)
cv.imshow("color", image)
found_bricks = BrickDetection.GetBricks(image,verbose=False)


#Assembly List (Params: Brick_list)
#----------------------------------------------------------------------
    #Run through assembly order and check brick list if it has the brick color and so on.
    #Returns: Order of bricks 

def GetOrder(blueprint,found_bricks):
    order = []
    for name in blueprint:
        for brick in found_bricks:
            if name == brick.color:
                order.append(brick)
                found_bricks.remove(brick)
                break
    return order

bart = ["Yellow","Red","Blue","Yellow"]
marge = ["Yellow","Green","Green","Yellow","Blue"]


bartOrder = GetOrder(bart,found_bricks)
margeOrder = GetOrder(marge,found_bricks)

print(robot.Pose())
#robot.MoveJ(brick_loc_workspace[0])
robot.setJoints(brick_loc_workspace[0])

"""
def mapping(x,y,d):
    xC = ((d[0]*x)+(d[1]*y)+d[2])/((d[6]*x)+(d[7]*y)+1)
    yC = ((d[3]*x)+(d[4]*y)+d[5])/((d[6]*x)+(d[7]*y)+1)
    return xC,yC


 """

#Run Robot Part (Params: Order of Bricks)
#----------------------------------------------------------------------
    #Move joints Pickup and Place
    #Complete


if cv.waitKey():
    RDK.Cam2D_Close()
