from myUtils import getImageFaceBox
import cv2
import os
import csv



faceCascadeXMLlist = [
    'C:\\Users\\Shahar\\Anaconda3\\envs\\py35\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml'
    , 'C:\\Users\\Shahar\\Anaconda3\\envs\\py35\\Library\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml'
    , 'C:\\Users\\Shahar\\Anaconda3\\envs\\py35\\Library\\etc\\haarcascades\\haarcascade_frontalface_alt_tree.xml'
    , 'C:\\Users\\Shahar\\Anaconda3\\envs\\py35\\Library\\etc\\haarcascades\\haarcascade_frontalface_alt.xml'
    , 'C:\\Users\\Shahar\\Anaconda3\\envs\\py35\\Library\\etc\\haarcascades\\haarcascade_frontalcatface_extended.xml'
    , 'C:\\Users\\Shahar\\Anaconda3\\envs\\py35\\Library\\etc\\haarcascades\\haarcascade_frontalcatface.xml'
    ]


def getFaceBoxDimension_faceCascade(filepath):
    image = cv2.imread(filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for casca in faceCascadeXMLlist:
        faceCascade = cv2.CascadeClassifier(casca)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            continue

        for (x, y, w, h) in faces:
            return x, y, w, h

    return -1, -1, -1, -1


inputDirectory = "C:\\Users\\Shahar\\Desktop\\PythonProjects\\ComputerVision\\Project\\HELEN"
outputDirectory = "C:\\Users\\Shahar\\Desktop\\PythonProjects\\ComputerVision\\Project\\FPN_MODEL_INPUT"

counter = 0

with open(outputDirectory + '\\input.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ID", "FILE", "FACE_X", "FACE_Y", "FACE_WIDTH", "FACE_HEIGHT"])
    for filename in os.listdir(inputDirectory):
        if filename.endswith(".jpg"):
            imPath = os.path.join(inputDirectory, filename)
            imPath = os.path.join(os.getcwd(), imPath)
            x, y, w, h = -1, -1, -1, -1

            try:
                x, y, w, h = getImageFaceBox(imPath)
            except:
                continue

            if w == -1:
                continue

            print(imPath)
            writer.writerow(["place_holder_" + str(counter), imPath, x, y, w, h])
            counter = counter + 1
