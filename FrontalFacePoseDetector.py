from myUtils import getImagePath68LandMarks,HELLEN_DIR,loadImages6DOF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tensorflow import keras
import pandas as pd
import numpy as np
import pickle
import os

ValidationDir = "./valid_set"


def getImagesNamesToLandMarksDict():
    imagesNameToLandMarksDict = dict()
    for imageName in os.listdir(HELLEN_DIR):
        if imageName.endswith(".jpg"):
            imageFilePath = os.path.join(HELLEN_DIR, imageName)
            imageFilePath = os.path.join(os.getcwd(), imageFilePath)
            im68Marks = getImagePath68LandMarks(imageFilePath)
            imagesNameToLandMarksDict[imageName] = im68Marks

    return imagesNameToLandMarksDict

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

def FacePoseDetctorBaseOnLandMarksDistance():

    imagesNameToLandMarks = None
    imageNameTo6DOF = None

    imagesLandMarksDistances = None
    images3DOF = None

    if os.path.isfile('./FrontalFacePoseDetector_pickles/imagesNameToLandMarks.pkl'):
        imagesNameToLandMarks = pickle.load(open("./FrontalFacePoseDetector_pickles/imagesNameToLandMarks.pkl", "rb"))
        imageNameTo6DOF = pickle.load(open("./FrontalFacePoseDetector_pickles/imageNameTo6DOF.pkl", "rb"))
    else:
        imageNameTo6DOF = loadImages6DOF()
        imagesNameToLandMarks = getImagesNamesToLandMarksDict()

        pickle.dump(imagesNameToLandMarks, open("./FrontalFacePoseDetector_pickles/imagesNameToLandMarks.pkl", "wb"))
        pickle.dump(imageNameTo6DOF, open("./FrontalFacePoseDetector_pickles/imageNameTo6DOF.pkl", "wb"))

    if os.path.isfile('./FrontalFacePoseDetector_pickles/imagesLandMarksDistances.pkl'):
        imagesLandMarksDistances = pickle.load(open("./FrontalFacePoseDetector_pickles/imagesLandMarksDistances.pkl", "rb"))
        images3DOF = pickle.load(open("./FrontalFacePoseDetector_pickles/images3DOF.pkl", "rb"))
    else:
        imagesLandMarksDistances = np.full((1, 2278), 0)
        images3DOF = np.full((1, 3), 0)

        for imageName in imagesNameToLandMarks:
            imageLandMark = imagesNameToLandMarks[imageName]

            if imageLandMark is None:
                continue

            try:
                image6dof = imageNameTo6DOF[imageName.replace("jpg", "txt")]
            except:
                continue

            imageLandMarkDistanceVector = getDistanceVectorForImageLandMarks(imageLandMark)
            imagesLandMarksDistances = np.concatenate((imagesLandMarksDistances, imageLandMarkDistanceVector), axis=0)

            image6dof = image6dof[:3]
            image6dof = np.array(image6dof)
            image6dof = np.reshape(image6dof, (1, len(image6dof)))
            images3DOF = np.concatenate((images3DOF, image6dof), axis=0)

        imagesLandMarksDistances = np.delete(imagesLandMarksDistances, 0, 0)
        images3DOF = np.delete(images3DOF, 0, 0)

        pickle.dump(imagesLandMarksDistances, open("./FrontalFacePoseDetector_pickles/imagesLandMarksDistances.pkl", "wb"))
        pickle.dump(images3DOF, open("./FrontalFacePoseDetector_pickles/images3DOF.pkl", "wb"))

    x_train, x_test, y_train, y_test = train_test_split(imagesLandMarksDistances, images3DOF, test_size=0.3, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

    std = StandardScaler()
    std.fit(x_train)
    x_train = std.transform(x_train)
    x_val = std.transform(x_val)
    x_test = std.transform(x_test)

    BATCH_SIZE = 62
    EPOCHS = 10000

    model = keras.models.Sequential()
    model.add(keras.layers.Dropout(0.2, input_shape=(imagesLandMarksDistances.shape[1],)))
    model.add(keras.layers.Dense(units=32, activation='relu', kernel_regularizer='l2'))
    model.add(keras.layers.Dense(units=16, activation='relu', kernel_regularizer='l2'))
    model.add(keras.layers.Dense(units=3, activation='linear'))

    fileName = "Model_epoc{epoch:02d}_loss{val_loss:.4f}_acc{val_acc:.2f}.h5"


    modelCheckPoints = keras.callbacks.ModelCheckpoint("./FrontalFacePoseDetector_pickles/Model/"+fileName,
                                                                 monitor='val_loss',
                                                                 verbose=1,
                                                                 save_best_only=True,
                                                                 save_weights_only=False,
                                                                 mode='auto',
                                                                 period=1)

    earlyStopCall = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=30)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                  factor=0.2,
                                                  patience=8,
                                                  min_lr=0.001)


    callback_list = [earlyStopCall, modelCheckPoints, reduce_lr]


    model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])

    hist = model.fit(x=x_train, y=y_train,
                     validation_data=(x_val, y_val),
                     batch_size=BATCH_SIZE,
                     epochs=EPOCHS,
                     callbacks=callback_list)


    pickle.dump(hist.history, open("./FrontalFacePoseDetector_pickles/FrontalFacePoseDetector_history.pkl", "wb"))

    print('Train loss:', model.evaluate(x_train, y_train, verbose=0))
    print('  Val loss:', model.evaluate(x_val, y_val, verbose=0))
    print(' Test loss:', model.evaluate(x_test, y_test, verbose=0))

    pd.DataFrame(hist.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()


if __name__ == "__main__":
    FacePoseDetctorBaseOnLandMarksDistance()















