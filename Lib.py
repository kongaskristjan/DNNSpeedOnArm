
import cv2
import ImageNetClasses

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


def printTopK(output, k=5):
    assert output.shape[1] == len(imageNetClasses) or output.shape[1] == len(imageNetClasses) + 1
    assert k <= len(imageNetClasses)

    labelsOffset = output.shape[1] - len(imageNetClasses)
    probAndIndex = []
    for i in range(len(imageNetClasses)):
        probAndIndex.append([output[0, i + labelsOffset], i])

    probAndIndex = list(reversed(sorted(probAndIndex)))
    for i in range(k):
        prob, index = probAndIndex[i]
        print("{0} ({1}): {2}%".format(indexToName(index), index, 100. * prob))


def indexToName(index):
    return imageNetClasses[index]


def nameToIndex(name):
    return transposedImageNetClasses[name]


def transposeDictionary(input):
    output = { input[key]: key for key in input }
    return output

imageNetClasses = ImageNetClasses.imageNetClasses
transposedImageNetClasses = transposeDictionary(imageNetClasses)
