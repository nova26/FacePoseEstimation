import sys
from tensorflow import keras
import cv2
import numpy as np
from myUtils import resizeImage,getImagePath68LandMarks,getDistanceVectorForImageLandMarks

import glob
import os
import csv


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if len(sys.argv) < 2:
    print("-E- Please pass a file with the input images")
    exit(1)

inputFile = sys.argv[1]


FrontalOrProfileFaceDetectorModel = keras.models.load_model("./FrontalOrProfileFaceDetector_pickles\\Model\\ModelIsProfile.h5")


ProfileFacePoseDetector = keras.models.load_model("./ProfileFacePoseDetector_pickles\\Model\\Model_epoc233_loss0.0819_acc0.77.h5")
FrontalFacePoseDetector =keras.models.load_model("./FrontalFacePoseDetector_pickles\\Model\\model.h5")
FrontalFacePoseDetectorCNN =keras.models.load_model("./FrontalFacePoseDetectorCNN_pickles\\Model\\Model_epoc02_loss0.0159_acc1.00.h5")

counter = 0

with open('headPoseOut.csv', 'w', newline='') as out_file:
    writer = csv.writer(out_file)
    writer.writerow(["file name", "rx", "ry", "rz"])
    with open(inputFile, 'r', newline='') as in_file:

        for line in in_file:
            print("-I- Processing "+line)

            line = line.strip()

            image = cv2.imread(line)
            image256 = resizeImage(image, 256)
            image250 = resizeImage(image, 250)
            image512 = resizeImage(image, 512)

            image = image[np.newaxis]
            image256 = image256[np.newaxis]
            image250 = image250[np.newaxis]
            image512 = image512[np.newaxis]

            isProfile = FrontalOrProfileFaceDetectorModel.predict(image512)[0][0]
            isProfile = int(round(isProfile))

            headPose = None

            image = image250

            im68Marks = getImagePath68LandMarks(line)

            if not isProfile or im68Marks is not None:

                if im68Marks is None:
                    headPose = FrontalFacePoseDetectorCNN.predict(image)
                else:
                    imageLandMarkDistanceVector = getDistanceVectorForImageLandMarks(im68Marks)
                    headPose2 = FrontalFacePoseDetectorCNN.predict(image)
                    headPose = FrontalFacePoseDetector.predict(imageLandMarkDistanceVector)
                    headPose = [[(headPose[0][0]+headPose2[0][0])/2,(headPose[0][1]+headPose2[0][1])/2,(headPose[0][2]+headPose2[0][2])/2]]
            else:
                headPose = ProfileFacePoseDetector.predict(image512)

            try:
                if '\\' in line:
                    line =line.split("\\")[-1]
                elif '/' in line:
                    line = line.split("/")[-1]
            except:
                pass

            writer.writerow([line, headPose[0][0], headPose[0][1], headPose[0][2]])
            print(str(line.split("\\")[-1])+str(headPose))
            counter+=1



