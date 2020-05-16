import numpy as np 
import cv2 as cv 
import glob 
from numpy import savetxt
import pandas
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.png')
print (images)
for fname in images:   
    img= cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,7), None)
    # If found, add object points, image points (after refining them)
    print(ret)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,7), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

cv.destroyAllWindows()

rets, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
savetxt('mtx.csv',mtx, delimiter=',')
print(rvecs[0])
rvecs0=cv.Rodrigues(rvecs[0])
print(rvecs0)

rvecs=np.array(rvecs)
print(rvecs.shape)
print(rvecs[0].shape)
print(rets)
''' rvecs=pandas.DataFrame(rvecs)
rvecs.to_csv('rvecs.csv')
print(type(rvecs))
print(rvecs)
#savetxt('rvecs.csv',rvecs,delimiter=',')

print(type(rvecs))
print(type(mtx)) '''
""" img = cv.imread('d:/test2.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst) """