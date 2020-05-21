
#mtx is the cameras intrinsic and extrinsic parameters
import numpy as np


def calibrate(brick_loc_image,brick_loc_workspace):
    #['YellowCalib','RedCalib','BlueCalib','GreenCalib']    
    
    # Adds the brick coordinates in correct order according to the colors
    # This is done to make sure that the correspond to the correct position in workspace
    for brick in brick_loc_image:
        if brick.color == "Yellow":
            x1 = brick.pickup_position[0]
            y1 = brick.pickup_position[1]
        if brick.color == "Red":
            x2 = brick.pickup_position[0]
            y2 = brick.pickup_position[1]
        if brick.color == "Blue":
            x3 = brick.pickup_position[0]
            y3 = brick.pickup_position[1]
        if brick.color == "Green":
            x4 = brick.pickup_position[0]
            y4 = brick.pickup_position[1]
    
    #yellow
    x1c = brick_loc_workspace[0].Cols()[3][0]
    y1c = brick_loc_workspace[0].Cols()[3][1]

    #Red
    x2c = brick_loc_workspace[1].Cols()[3][0]
    y2c = brick_loc_workspace[1].Cols()[3][1]
    #Blue
    x3c = brick_loc_workspace[2].Cols()[3][0]
    y3c = brick_loc_workspace[2].Cols()[3][1]
    #Green
    x4c = brick_loc_workspace[3].Cols()[3][0]
    y4c = brick_loc_workspace[3].Cols()[3][1]
    
    #Here the matrix with equations is setup
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

    #Here the other site of the equations is setup
    b = [x1c,y1c,x2c,y2c,x3c,y3c,x4c,y4c]

    #Here the array of d is found by solving the linear system
    d=np.linalg.solve(a,b)


    return d