
import random
import numpy as np
import cv2
import pickle
import os
from tensorflow import keras
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd

from myUtils import getImageNumberOfFaces


def trainFrontalOrFacialDetector():
    import glob
    profileImagesToTest = glob.glob("./data\\profile_images/*.jpg")

    frontalImagesToTest = glob.glob("./data\\HELEN/*.jpg")
    frontalImagesToTest = glob.glob("./data\\test_set/*.png")


    IMAGE_SIZE = 512

    imageSetimagesPath = []

    if not os.path.isfile('./FrontalOrProfileFaceDetector_pickles/imageSetimagesPath.pkl'):
        print("Building imageSetimagesPath ...")

        count = 1
        totalImagesToLoad = 368

        for profileFace in profileImagesToTest:
            imageSetimagesPath.append(profileFace)
            if len(imageSetimagesPath) == totalImagesToLoad:
                break
            print("profileFace Loade {0}/{1}".format(count,totalImagesToLoad))
            count = count +1

        for frontalFace in frontalImagesToTest:
            numberOfFaces = getImageNumberOfFaces(frontalFace)
            if numberOfFaces != 1 :
                continue
            imageSetimagesPath.append(frontalFace)
            if len(imageSetimagesPath) == 10000:
                break

            print("frontalFace loade {0}/{1}".format(count,totalImagesToLoad))
            count = count + 1

        pickle.dump(imageSetimagesPath, open("./FrontalOrProfileFaceDetector_pickles/imageSetimagesPath.pkl", "wb"))
    else:
        imageSetimagesPath = pickle.load(open("./FrontalOrProfileFaceDetector_pickles/imageSetimagesPath.pkl", "rb"))

    if not os.path.isfile("./FrontalOrProfileFaceDetector_pickles/ImageSet/imageSet{0}.pkl".format(200)):

        imageSet = np.full((1, IMAGE_SIZE, IMAGE_SIZE, 3), 0)
        imagesLabels = []

        totalImages = len(imageSetimagesPath)
        index = 1

        for im in imageSetimagesPath:
            print("images {0}/{1}".format(index,totalImages))


            image = cv2.imread(im)

            if image.shape[1] > IMAGE_SIZE:
                image = cv2.resize(image, (IMAGE_SIZE, image.shape[0]), interpolation=cv2.INTER_AREA)

            if image.shape[0] > IMAGE_SIZE:
                image = cv2.resize(image, (image.shape[1], IMAGE_SIZE), interpolation=cv2.INTER_AREA)

            heightPaddint = IMAGE_SIZE - image.shape[0]
            heightPaddintTop = int(heightPaddint / 2)
            heightPaddintButton = heightPaddintTop

            if heightPaddintTop + heightPaddintButton != heightPaddint and heightPaddint:
                heightPaddintTop = heightPaddintTop + 1

            widthPadding = IMAGE_SIZE - image.shape[1]
            widthPaddingRight = int(widthPadding / 2)
            widthPaddingLeft = widthPaddingRight

            if widthPaddingRight + widthPaddingLeft != widthPadding and widthPadding:
                widthPaddingRight = widthPaddingRight + 1

            image = image / 255
            image = cv2.copyMakeBorder(image.copy(), heightPaddintTop, heightPaddintButton, widthPaddingRight,
                                       widthPaddingLeft, cv2.BORDER_CONSTANT, value=[0, 0, 0])

            image = np.reshape(image, (1, IMAGE_SIZE, IMAGE_SIZE, 3))
            imageSet = np.concatenate((imageSet, image), axis=0)

            if 'profile_images' in im:
                imagesLabels.append(0)
            else:
                imagesLabels.append(1)

            if index % 200 == 0:
                imageSet = np.delete(imageSet, 0, 0)

                pickle.dump(imageSet, open("./FrontalOrProfileFaceDetector_pickles/imageSet{0}.pkl".format(index), "wb"))
                pickle.dump(imagesLabels, open("./FrontalOrProfileFaceDetector_pickles/imagesLabels{0}.pkl".format(index), "wb"))

                imageSet = np.full((1, IMAGE_SIZE, IMAGE_SIZE, 3), 0)
                imagesLabels = []

            index = index + 1


    print(200)
    imageSet = pickle.load(open("./FrontalOrProfileFaceDetector_pickles/ImageSet/imageSet{0}.pkl".format(200), "rb"))
    imagesLabels = pickle.load(open("./FrontalOrProfileFaceDetector_pickles/ImageSet/imagesLabels{0}.pkl".format(200), "rb"))

    for x in range(400,10200,200):



        imageSetTmp = pickle.load(open("./FrontalOrProfileFaceDetector_pickles/ImageSet/imageSet{0}.pkl".format(x), "rb"))
        imagesLabelsTmp = pickle.load(open("./FrontalOrProfileFaceDetector_pickles/ImageSet/imagesLabels{0}.pkl".format(x), "rb"))
        imageSet = np.concatenate((imageSet, imageSetTmp), axis=0)
        imagesLabels = imagesLabels + imagesLabelsTmp
        print(x)



    DefaultConv2D = partial(keras.layers.Conv2D,kernel_size=3, activation='relu', padding="SAME")

    model = keras.models.Sequential([
        DefaultConv2D(filters=32, kernel_size=3, input_shape=[IMAGE_SIZE, IMAGE_SIZE, 3]),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=2),
        DefaultConv2D(filters=128),
        DefaultConv2D(filters=128),
        keras.layers.MaxPooling2D(pool_size=2),
        DefaultConv2D(filters=128),
        DefaultConv2D(filters=128),
        keras.layers.MaxPooling2D(pool_size=2),
        DefaultConv2D(filters=256),
        DefaultConv2D(filters=256),
        keras.layers.MaxPooling2D(pool_size=2),
        DefaultConv2D(filters=256),
        DefaultConv2D(filters=256),
        keras.layers.MaxPooling2D(pool_size=2),
        DefaultConv2D(filters=256),
        DefaultConv2D(filters=384),
        DefaultConv2D(filters=384),
        keras.layers.MaxPooling2D(pool_size=2),
        DefaultConv2D(filters=384),
        DefaultConv2D(filters=384),
        DefaultConv2D(filters=512),
        DefaultConv2D(filters=512),
        DefaultConv2D(filters=512),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(units=32, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(units=16, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(units=1, activation='sigmoid')
    ])

    fileName = "Model_epoc{epoch:02d}_loss{val_loss:.4f}_acc{val_acc:.2f}.h5"

    modelCheckPoints = keras.callbacks.ModelCheckpoint("./FrontalOrProfileFaceDetector_pickles/Model/"+fileName,
                                                                 monitor='val_loss',
                                                                 verbose=1,
                                                                 save_best_only=True,
                                                                 save_weights_only=False,
                                                                 mode='auto',
                                                                 period=1)

    earlyStopCall = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=15)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                  patience=5,
                                                  factor=0.2,
                                                  cooldown=3,
                                                  verbose=1,
                                                  min_lr=0.001)

    callback_list = [earlyStopCall, modelCheckPoints, reduce_lr]

    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)


    model.compile(optimizer=sgd,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(imageSet, imagesLabels,
                        validation_split=0.33,
                        callbacks=callback_list,
                        epochs=1000,
                        batch_size=64,
                        shuffle=True,
                        verbose=1)

    model.save("./FrontalOrProfileFaceDetector_pickles/FrontalOrProfileFaceDetector.h5")

    pickle.dump(history.history, open("./FrontalOrProfileFaceDetector_pickles/FrontalOrProfileFaceDetector_history.pkl", "wb"))

    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()









trainFrontalOrFacialDetector()