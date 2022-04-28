# System information:
# - Linux Mint 18.1 Cinnamon 64-bit
# - Python 2.7 with OpenCV 3.2.0

import numpy
import cv2
from cv2 import aruco
import pickle
import glob
import numpy as np
import ConfigParser


# ChAruco board variables
CHARUCOBOARD_ROWCOUNT = 7	
CHARUCOBOARD_COLCOUNT = 5
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_100)

# Create constants to be passed into OpenCV and Aruco methods
CHARUCO_BOARD = aruco.CharucoBoard_create(
        squaresX=CHARUCOBOARD_COLCOUNT,
        squaresY=CHARUCOBOARD_ROWCOUNT,
        squareLength=0.0361,
        markerLength=0.0244,
        dictionary=ARUCO_DICT)
markerLength = 0.0244
color_width = 1920
color_height = 1080
camera_info_path = '/home/andy/aruco_ws/src/ArucoMark/CalibrationByCharucoBoard/camera_8_internal.ini'

config = ConfigParser.ConfigParser()
config.optionxform = str            
config.read(camera_info_path)

internal_name = 'Internal_' + str(color_width) + '_' + str(color_height)
b00 = float(config.get(internal_name, "Key_1_1"))
b01 = float(config.get(internal_name, "Key_1_2"))
b02 = float(config.get(internal_name, "Key_1_3"))
b10 = float(config.get(internal_name, "Key_2_1"))
b11 = float(config.get(internal_name, "Key_2_2"))
b12 = float(config.get(internal_name, "Key_2_3"))
b20 = float(config.get(internal_name, "Key_3_1"))
b21 = float(config.get(internal_name, "Key_3_2"))
b22 = float(config.get(internal_name, "Key_3_3"))

intrinsic_matrix = np.mat([[b00, b01, b02],
                           [b10, b11, b12],
                           [b20, b21, b22]])

distcoeff_name = 'DistCoeffs_' + str(color_width) + '_' + str(color_height)
k_1 = float(config.get(distcoeff_name, "K_1"))
k_2 = float(config.get(distcoeff_name, "K_2"))
k_3 = float(config.get(distcoeff_name, "K_3"))
p_1 = float(config.get(distcoeff_name, "p_1"))
p_2 = float(config.get(distcoeff_name, "p_2"))   

distortion_coeff = np.array([k_1, k_2, p_1, p_2, k_3]) 
# distortion_coeff = np.array([0.0, 0, 0, 0, 0])

# Create the arrays and variables we'll use to store info like corners and IDs from images processed
corners_all = [] # Corners discovered in all images processed
ids_all = [] # Aruco ids corresponding to corners discovered
image_size = None # Determined at runtime


# This requires a set of images or a video taken with the camera you want to calibrate
# I'm using a set of images taken with the camera with the naming convention:
# 'camera-pic-of-charucoboard-<NUMBER>.jpg'
# All images used should be the same size, which if taken with the same camera shouldn't be a problem
images = glob.glob('./pic_test/camera-pic-of-charucoboard-*.jpg')

# Loop through images glob'ed
for iname in images:
    # Open the image
    img = cv2.imread(iname)
    # Grayscale the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find aruco markers in the query image
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
            image=gray,
            dictionary=ARUCO_DICT)
    rvecs, tvecs = aruco.estimatePoseSingleMarkers(corners, markerLength, intrinsic_matrix, distortion_coeff)
    
    print('ids by detecMarkers', np.shape(ids))
    print(ids)
    print('corners by detecMarkers', np.shape(corners))
    print(corners)
    print('rvecs by detecMarkers', np.shape(rvecs))
    print(rvecs)
    print('tvecs by detecMarkers', np.shape(tvecs))
    print(tvecs)

    # Outline the aruco markers found in our query image
    img = aruco.drawDetectedMarkers(
            image=img, 
            corners=corners)

    corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(
            image = gray,
            board = CHARUCO_BOARD,
            detectedCorners = corners,
            detectedIds = ids,
            rejectedCorners = rejectedImgPoints,
            cameraMatrix = intrinsic_matrix,
            distCoeffs = distortion_coeff)

    # Get charuco corners and ids from detected aruco markers
    response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=CHARUCO_BOARD)
    pose, rvec, tvec = aruco.estimatePoseCharucoBoard(
                        charucoCorners=charuco_corners, 
                        charucoIds=charuco_ids, 
                        board=CHARUCO_BOARD, 
                        cameraMatrix=intrinsic_matrix, 
                        distCoeffs=distortion_coeff)
    print('charuco_ids by CharucoBoard: ', np.shape(charuco_ids))
    print(charuco_ids)
    print('corners by CharucoBoard: ', np.shape(charuco_corners))
    print(charuco_corners)
    print('rvecs by CharucoBoard: ', np.shape(rvec))
    print(rvec)
    print('tvecs by CharucoBoard: ', np.shape(tvec))
    print(tvec)

    rvecs = []
    tvecs = []
    ch_ids = []
    rvecs.append(np.array(rvec).reshape(1, 3))
    tvecs.append(np.array(tvec).reshape(1, 3))
    ch_ids.append([0])
    [corners[0]]

    print('charuco_ids by reshape: ', np.shape(ch_ids))
    print('corners by reshape: ', np.shape([corners[0]]))
    print('rvecs by reshape: ', np.shape(rvecs))
    print('tvecs by reshape: ', np.shape(tvecs))

    # If a Charuco board was found, let's collect image/corner points
    # Requiring at least 20 squares
    if response > 20:
        # Add these corners and ids to our calibration arrays
        corners_all.append(charuco_corners)
        ids_all.append(charuco_ids)
        
        # Draw the Charuco board we've detected to show our calibrator the board was properly detected
        img = aruco.drawDetectedCornersCharuco(
                image=img,
                charucoCorners=charuco_corners,
                charucoIds=charuco_ids)
       
        # If our image size is unknown, set it now
        if not image_size:
            image_size = gray.shape[::-1]
    
        # Reproportion the image, maxing width or height at 1000
        proportion = max(img.shape) / 1280.0
        img = cv2.resize(img, (int(img.shape[1]/proportion), int(img.shape[0]/proportion)))
        # Pause to display each image, waiting for key press
        cv2.imshow('Charuco board', img)
        cv2.waitKey(0)
    else:
        print("Not able to detect a charuco board in image: {}".format(iname))

# Destroy any open CV windows
cv2.destroyAllWindows()

# Make sure at least one image was found
if len(images) < 1:
    # Calibration failed because there were no images, warn the user
    print("Calibration was unsuccessful. No images of charucoboards were found. Add images of charucoboards and use or alter the naming conventions used in this file.")
    # Exit for failure
    exit()

# Make sure we were able to calibrate on at least one charucoboard by checking
# if we ever determined the image size
if not image_size:
    # Calibration failed because we didn't see any charucoboards of the PatternSize used
    print("Calibration was unsuccessful. We couldn't detect charucoboards in any of the images supplied. Try changing the patternSize passed into Charucoboard_create(), or try different pictures of charucoboards.")
    # Exit for failure
    exit()

# Now that we've seen all of our images, perform the camera calibration
# based on the set of points we've discovered
calibration, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=corners_all,
        charucoIds=ids_all,
        board=CHARUCO_BOARD,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None)
    
# Print matrix and distortion coefficient to the console
print(cameraMatrix)
print(distCoeffs)
    
# Save values to be used where matrix+dist is required, for instance for posture estimation
# I save files in a pickle file, but you can use yaml or whatever works for you
f = open('calibration.pckl', 'wb')
pickle.dump((cameraMatrix, distCoeffs, rvecs, tvecs), f)
f.close()
    
# Print to console our success
print('Calibration successful. Calibration file used: {}'.format('calibration.pckl'))

