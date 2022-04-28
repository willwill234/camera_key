import cv2
import cv2.aruco as aruco

# Create ChArUco board, which is a set of Aruco markers in a chessboard setting
# meant for calibration
# the following call gets a ChArUco board of tiles 5 wide X 7 tall
gridboard = aruco.CharucoBoard_create(
        squaresX=3, 
        squaresY=5, 
        squareLength=0.035,
        markerLength=0.025, 
        dictionary=aruco.Dictionary_get(aruco.DICT_4X4_50))

# Create an image from the gridboard
# img = gridboard.draw(outSize=(988, 1400))
img = gridboard.draw(outSize=(700, 1100))
cv2.imwrite("3x5_charuco.jpg", img)

# Display the image to us
cv2.imshow('Gridboard', img)
# Exit on any key
cv2.waitKey(0)
cv2.destroyAllWindows()

