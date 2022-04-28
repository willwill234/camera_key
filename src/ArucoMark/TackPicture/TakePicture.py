import pyrealsense2 as rs
import numpy as np
import cv2
X = 1920
Y = 1080
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, X, Y, rs.format.bgr8, 30)

pipeline.start(config)

#cap = cv2.VideoCapture(4)
frameId = 0
while(True):

    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())


    
    # Capture frame-by-frame
    # ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    show_image = cv2.resize(color_image, (X/2, Y/2))                # Resize image
    cv2.imshow('frame',show_image)

    key = cv2.waitKey(10)

    if key & 0xFF == ord('t'):
        filename = "./pic/camera-pic-of-charucoboard-" +  str(int(frameId)) + ".jpg"
        cv2.imwrite(filename, color_image)
        frameId += 1
        cv2.waitKey(100)

    if key & 0xFF == ord('q'):
        break

# When everything done, release the capture
# cap.release()
cv2.destroyAllWindows()
# cv2.waitKey(1) & 
