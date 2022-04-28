# The following code is used to watch a video stream, detect Aruco markers, and use
# a set of markers to determine the posture of the camera in relation to the plane
# of markers.
#
# Assumes that all markers are on the same plane, for example on the same piece of paper
#
# Requires camera calibration (see the rest of the project for example calibration)

import rospy
import numpy as np
import cv2
import cv2.aruco as aruco
import os
import pickle

NUMBER = 50
# Constant parameters used in Aruco methods
ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_100)

class MarkerPosture():
    def __init__(self):
        # Check for camera calibration data
        if not os.path.exists('./calibration.pckl'):
            print("You need to calibrate the camera you'll be using. See calibration project directory for details.")
            exit()
        else:
            f = open('calibration.pckl', 'rb')
            (self.cameraMatrix, self.distCoeffs, _, _) = pickle.load(f)
            f.close()
            if self.cameraMatrix is None or self.distCoeffs is None:
                print("Calibration issue. Remove ./calibration.pckl and recalibrate your camera with CalibrateCamera.py.")
                exit()

        # Create grid board object we're using in our stream
        # board = aruco.GridBoard_create(
        #         markersX=2,
        #         markersY=2,
        #         markerLength=0.09,
        #         markerSeparation=0.01,
        #         dictionary=ARUCO_DICT)

        # Create vectors we'll be using for rotations and translations for postures
        self.rvecs = None 
        self.tvecs = None
        self.rvecs_arr = np.zeros((100, 3, NUMBER))
        self.tvecs_arr = np.zeros((100, 3, NUMBER))

        # cam = cv2.VideoCapture('gridboardiphonetest.mp4')
        self.cam = cv2.VideoCapture(2)

    def findMarker(self):
        self.rvecs_arr = np.zeros((100, 3, NUMBER))
        self.tvecs_arr = np.zeros((100, 3, NUMBER))
        QueryImg = None
        for order in range (NUMBER):
            # Capturing each frame of our video stream
            ret, QueryImg = self.cam.read()
            if ret == True:
                # grayscale image
                gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)

                # Detect Aruco markers
                corners, ids, _ = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)

                # Refine detected markers
                # Eliminates markers not part of our board, adds missing markers to the board
                # corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(
                #         image = gray,
                #         board = board,
                #         detectedCorners = corners,
                #         detectedIds = ids,
                #         rejectedCorners = rejectedImgPoints,
                #         self.cameraMatrix = self.cameraMatrix,
                #         self.distCoeffs = self.distCoeffs)   

                ###########################################################################
                # TODO: Add validation here to reject IDs/corners not part of a gridboard #
                ###########################################################################

                # Outline all of the markers detected in our image
                QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, borderColor=(0, 0, 255))

                # Require 15 markers before drawing axis
                if ids is not None and len(ids) > 0:
                    # Estimate the posture of the gridboard, which is a construction of 3D space based on the 2D video 
                    #pose, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, self.cameraMatrix, self.distCoeffs)
                    #if pose:
                    #    # Draw the camera posture calculated from the gridboard
                    #    QueryImg = aruco.drawAxis(QueryImg, self.cameraMatrix, self.distCoeffs, rvec, tvec, 0.3)
                    # Estimate the posture per each Aruco marker
                    self.rvecs, self.tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.02, self.cameraMatrix, self.distCoeffs)           
                    for _id, rvec, tvec in zip(ids, self.rvecs, self.tvecs):
                        _id = _id[0]
                        for i in range(3):
                            self.rvecs_arr[_id][i][order] = rvec[0][i]
                            self.tvecs_arr[_id][i][order] = tvec[0][i]
                        
                    #     QueryImg = aruco.drawAxis(QueryImg, self.cameraMatrix, self.distCoeffs, rvec, tvec, 0.02)

                # Display our image
                # cv2.imshow('QueryImage', QueryImg)
            cv2.waitKey(1)
        # print('self.rvecs_arr = ', self.rvecs_arr)
        # print('self.tvecs_arr = ', self.tvecs_arr)
        result = np.array(())
        r_avg = np.zeros(3) 
        t_avg = np.zeros(3)
        for _id in range(100):
            ra = self.rvecs_arr[_id][0].nonzero()
            if len(ra[0]) < 0.3*NUMBER:
                continue
            rb = self.rvecs_arr[_id][1].nonzero()
            rc = self.rvecs_arr[_id][2].nonzero()
            tx = self.tvecs_arr[_id][0].nonzero()
            ty = self.tvecs_arr[_id][1].nonzero()
            tz = self.tvecs_arr[_id][2].nonzero()
            ra = self.rvecs_arr[_id][0][ra]
            rb = self.rvecs_arr[_id][1][rb]
            rc = self.rvecs_arr[_id][2][rc]
            tx = self.tvecs_arr[_id][0][tx]
            ty = self.tvecs_arr[_id][1][ty]
            tz = self.tvecs_arr[_id][2][tz]
            ra = np.sort(ra, kind = 'quicksort')
            rb = np.sort(rb, kind = 'quicksort')
            rc = np.sort(rc, kind = 'quicksort')
            tx = np.sort(tx, kind = 'quicksort')
            ty = np.sort(ty, kind = 'quicksort')
            tz = np.sort(tz, kind = 'quicksort')
            r = np.array((ra, rb, rc))
            t = np.array((tx, ty, tz))
            ctn = False
            for i in range(3):
                rv, tv = r[i], t[i]
                
                while np.std(rv) > 0.01 and len(rv) >= NUMBER*0.2:
                    if abs(rv[0] - np.average(rv)) > abs(rv[-1] - np.average(rv)):
                        rv = np.delete(rv, 0)
                    else:
                        rv = np.delete(rv, -1)
                while np.std(tv) > 0.01 and len(tv) >= NUMBER*0.2:
                    if abs(tv[0] - np.average(tv)) > abs(tv[-1] - np.average(tv)):
                        tv = np.delete(tv, 0)
                    else:
                        tv = np.delete(tv, -1)
                if len(rv) < NUMBER*0.2 or len(tv) < NUMBER*0.2:
                    ctn = True
                    break
                r_avg[i] = np.average(rv)
                t_avg[i] = np.average(tv)
            if ctn:
                continue
            # print('[_id, r,t] = ', [_id, r,t])
            result = np.append(result, [_id, np.copy(r_avg), np.copy(t_avg)])
        
        result = result.reshape(int(len(result)/3),3)
        # print(result)
        for rst in result:
            QueryImg = aruco.drawAxis(QueryImg, self.cameraMatrix, self.distCoeffs, rst[1], rst[2], 0.02)
        cv2.imshow('QueryImage', QueryImg)
        return result
            
if __name__ == '__main__':
    rospy.init_node('aruco')
    mp = MarkerPosture()
    while True:
        result = mp.findMarker()
        print(result)
        print(cv2.Rodrigues(result[0][1])[0])
        # print('==========================')
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

        


# cv2.destroyAllWindows()
