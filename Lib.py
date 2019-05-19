
import cv2

def readImage(fName):
    image = cv2.imread(fName, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def display(imageRGB):
    imageBGR = cv2.cvtColor(imageRGB, cv2.COLOR_RGB2BGR)
    cv2.namedWindow("", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("", imageBGR)
    while cv2.waitKey(0) == ord('q'):
        pass
