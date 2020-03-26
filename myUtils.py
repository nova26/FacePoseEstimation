import dlib
import cv2
import imutils
from imutils import face_utils
import os
import collections
import numpy as np

Image6DOF = collections.namedtuple('Image6DOF', ['pitch', 'yaw', 'roll', 'tx', 'ty', 'tz'])

myUtils_shape68 = "./data\\shape_predictor_68_face_landmarks\\shape_predictor_68_face_landmarks.dat"


HELLEN_DIR = "./data\\HELEN"

myUtils_detector = dlib.get_frontal_face_detector()
myUtils_predictor = dlib.shape_predictor(myUtils_shape68)


def getImageFaceBox(imPath):
    detector = dlib.get_frontal_face_detector()
    image = cv2.imread(imPath)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        return x, y, w, h
    return -1, -1, -1, -1


def get7faceLandMarks(imPath):
    detector = dlib.get_frontal_face_detector()
    image = cv2.imread(imPath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = myUtils_predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        face6LandMarks = np.full((1, 2), 0, dtype="double")
        face6LandMarks = np.concatenate((face6LandMarks, np.reshape(shape[33], (1, 2))), axis=0)  # noseTip
        face6LandMarks = np.concatenate((face6LandMarks, np.reshape(shape[8], (1, 2))), axis=0)  # chin
        face6LandMarks = np.concatenate((face6LandMarks, np.reshape(shape[36], (1, 2))), axis=0)  # leftEyeCorner
        face6LandMarks = np.concatenate((face6LandMarks, np.reshape(shape[45], (1, 2))), axis=0)  # rightEyeCorner
        face6LandMarks = np.concatenate((face6LandMarks, np.reshape(shape[49], (1, 2))), axis=0)  # leftMouth
        face6LandMarks = np.concatenate((face6LandMarks, np.reshape(shape[54], (1, 2))), axis=0)  # rightMouth
        face6LandMarks = np.delete(face6LandMarks, 0, 0)

        # (x, y, w, h) = face_utils.rect_to_bb(rect)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # for pont in face6LandMarks:
        #     cv2.circle(image, (int(pont[0]), int(pont[1])), 1, (0, 0, 255), -1)
        # cv2.imshow("Output", image)
        # cv2.waitKey(0)

        return face6LandMarks


def getImage68LandMarks(image):
#    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = myUtils_detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = myUtils_predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        return shape


def getImagePath68LandMarks(imPath):
    image = cv2.imread(imPath)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = myUtils_detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = myUtils_predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        return shape


def loadImages6DOF():
    imagesTo6DOF = {}
    imagesDOFDir = "./data\FPN_HELLEN_RESULTS"
    for filename in os.listdir(imagesDOFDir):
        if filename.endswith(".txt"):
            fileLines = []
            with open(imagesDOFDir + "\\" + filename, 'r') as outfile:
                fileLines = [line.replace('\n', '') for line in outfile]
            fileLines.append(0)
            fileLines = fileLines[1:]
            image6DOF = Image6DOF._make(fileLines)
            imagesTo6DOF[filename] = image6DOF
    return imagesTo6DOF

def getImageNumberOfFaces(imPath):
    detector = dlib.get_frontal_face_detector()
    image = cv2.imread(imPath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    return len(rects)

def resizeImage(image,freshSize):
    if image.shape[1] > freshSize:
        image = cv2.resize(image, (freshSize, image.shape[0]), interpolation=cv2.INTER_AREA)

    if image.shape[0] > freshSize:
        image = cv2.resize(image, (image.shape[1], freshSize), interpolation=cv2.INTER_AREA)

    heightPaddint = freshSize - image.shape[0]
    heightPaddintTop = int(heightPaddint / 2)
    heightPaddintButton = heightPaddintTop

    if heightPaddintTop + heightPaddintButton != heightPaddint and heightPaddint:
        heightPaddintTop = heightPaddintTop + 1

    widthPadding = freshSize - image.shape[1]
    widthPaddingRight = int(widthPadding / 2)
    widthPaddingLeft = widthPaddingRight

    if widthPaddingRight + widthPaddingLeft != widthPadding and widthPadding:
        widthPaddingRight = widthPaddingRight + 1

    image = image / 255
    image = cv2.copyMakeBorder(image.copy(), heightPaddintTop, heightPaddintButton, widthPaddingRight,
                               widthPaddingLeft, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return  image



def getDistanceVectorForImageLandMarks(imageLandMark):
    landMarksDistances = []
    for x in range(imageLandMark.shape[0]):
        for y in range(x+1,imageLandMark.shape[0]):
            distance = (imageLandMark[x][0] - imageLandMark[y][0])**2 + (imageLandMark[x][1] - imageLandMark[y][1])**2
            distance = distance**0.5
            landMarksDistances.append(distance)
    landMarksDistances  = np.array(landMarksDistances)
    landMarksDistances = np.reshape(landMarksDistances,(1,len(landMarksDistances)))
    return landMarksDistances














