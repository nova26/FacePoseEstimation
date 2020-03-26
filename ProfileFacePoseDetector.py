import cv2
from sklearn.model_selection import train_test_split


def copyFromTo():
    import glob
    import shutil
    import os
    fromFolder = './data\profile_images/*txt'
    tofolder = './data\FPN_PROFILE_RESULTS'

    folderFiles = glob.glob(fromFolder)

    for file in folderFiles:
        shutil.copyfile(file, tofolder + "/" + file.split("\\")[-1])
        os.remove(file)


copyFromTo()


def wrtieInputFileForFPNModel():
    import csv
    import glob
    counter = 0
    outputDirectory = "C:\\Users\\Shahar\\Desktop\\PythonProjects\\ComputerVision\\Project\\FPN_MODEL_INPUT_PROFILE"
    imagesToCopy = glob.glob(
        "C:\\Users\\Shahar\\Desktop\\PythonProjects\\ComputerVision\\Project\\profile_images/*.jpg")
    with open(outputDirectory + '\\input.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "FILE", "FACE_X", "FACE_Y", "FACE_WIDTH", "FACE_HEIGHT"])
        for filename in imagesToCopy:
            image = cv2.imread(filename)
            x = 1
            y = 1
            w = image.shape[0] - 1
            h = image.shape[1] - 1
            print(filename)

            writer.writerow(["place_holder_" + str(counter), filename, x, y, w, h])
            counter = counter + 1


def testProfile():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from tensorflow import keras
    import pickle
    from functools import partial
    import glob
    from random import randint, seed
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    seed(1)

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    IMAGE_SIZE = 512
    MAX_IMAGES_TO_LOAD = 5000

    profileImagesToTest = glob.glob("./data\\profile_images/*.jpg")
    profileImagesResults = glob.glob("./data\\FPN_PROFILE_RESULTS/*.txt")

    imageNameTo3DOFGround = {}

    if os.path.isfile('./ProfileFacePoseDetector_pickles/imageNameTo3DOFGround.pkl'):
        imageNameTo3DOFGround = pickle.load(open("./ProfileFacePoseDetector_pickles/imageNameTo3DOFGround.pkl", "rb"))
    else:
        count = 1
        size = len(profileImagesResults)
        for file in profileImagesResults:
            if count == MAX_IMAGES_TO_LOAD:
                break
            print("Ground true {0}/{1}".format(count, size))
            count = count + 1
            fileLines = []
            with open(file, 'r') as outfile:
                fileLines = [line.replace('\n', '') for line in outfile]
            fileLines.append(0)
            fileLines = np.array(fileLines[1:4])
            fileLines = np.reshape(fileLines, (1, -1))
            file = file.replace(".txt", "")
            file = file[file.rfind('\\') + 1:]
            imageNameTo3DOFGround[file] = fileLines

        pickle.dump(imageNameTo3DOFGround, open("./ProfileFacePoseDetector_pickles/imageNameTo3DOFGround.pkl", "wb"))

    profileImages = np.full((1, IMAGE_SIZE, IMAGE_SIZE, 3), 0)
    profileDOFlabels = np.full((1, 3), 0)

    if not os.path.isfile('./ProfileFacePoseDetector_pickles/500/profileImages_200.pkl'):

        x = 1
        s = len(profileImagesToTest)

        numberOfImagesToLoad = 10001

        groundImagesNames = imageNameTo3DOFGround.keys()

        for im in profileImagesToTest:
            n = im.replace(".jpg", "")
            n = n[n.rfind('\\') + 1:]

            if n not in groundImagesNames:
                continue
            if x == numberOfImagesToLoad:
                break

            print(im)
            print("{0}/{1}".format(x, s))

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

            image = cv2.copyMakeBorder(image.copy(), heightPaddintTop, heightPaddintButton, widthPaddingRight,
                                       widthPaddingLeft, cv2.BORDER_CONSTANT, value=[0, 0, 0])

            image = np.reshape(image, (1, IMAGE_SIZE, IMAGE_SIZE, 3))

            profileImages = np.concatenate((profileImages, image), axis=0)

            file = im.replace(".jpg", "")
            file = file[file.rfind('\\') + 1:]

            dof = imageNameTo3DOFGround[file]
            dof = np.reshape(dof, (1, 3))

            profileDOFlabels = np.concatenate((profileDOFlabels, dof), axis=0)

            if x % 200 == 0:
                profileImages = np.delete(profileImages, 0, 0)
                profileDOFlabels = np.delete(profileDOFlabels, 0, 0)

                pickle.dump(profileImages,
                            open("./ProfileFacePoseDetector_pickles/500/profileImages_{0}.pkl".format(x), "wb"))
                pickle.dump(profileDOFlabels,
                            open("./ProfileFacePoseDetector_pickles/500/profileDOFlabels_{0}.pkl".format(x), "wb"))

                profileImages = np.full((1, IMAGE_SIZE, IMAGE_SIZE, 3), 0)
                profileDOFlabels = np.full((1, 3), 0)

            x = x + 1

    print(200)
    profileImages = pickle.load(open("./ProfileFacePoseDetector_pickles/500/profileImages_{0}.pkl".format(200), "rb"))
    profileDOFlabels = pickle.load(
        open("./ProfileFacePoseDetector_pickles/500/profileDOFlabels_{0}.pkl".format(200), "rb"))

    for x in range(400, 10200, 200):
        profileImagesTmp = pickle.load(
            open("./ProfileFacePoseDetector_pickles/500/profileImages_{0}.pkl".format(x), "rb"))
        profileDOFlabelsTmp = pickle.load(
            open("./ProfileFacePoseDetector_pickles/500/profileDOFlabels_{0}.pkl".format(x), "rb"))

        profileImages = np.concatenate((profileImages, profileImagesTmp), axis=0)
        profileDOFlabels = np.concatenate((profileDOFlabels, profileDOFlabelsTmp), axis=0)
        print(x)



    profileImages = profileImages.astype(np.uint8)


    print("Done load all images")

    # profileImagesMean = profileImages.mean()
    # profileImagesStd = profileImages.std() + 1e-8
    #
    # profileImagesMean = profileImagesMean.astype('int32')
    # profileImagesStd = profileImagesStd.astype('int32')
    #
    # profileImagesStd-=profileImagesMean
    # profileImagesStd/=profileImagesStd



    DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3,
                            activation='relu',
                            padding="SAME"
                     #       kernel_initializer='he_normal'
                            )

    model = keras.models.Sequential([
        DefaultConv2D(filters=32, kernel_size=3, input_shape=[IMAGE_SIZE, IMAGE_SIZE, 3]),
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
        keras.layers.Dense(units=3, activation='linear')
    ])

    fileName = "Model_epoc{epoch:02d}_loss{val_loss:.4f}_acc{val_acc:.2f}.h5"

    modelCheckPoints = keras.callbacks.ModelCheckpoint("./ProfileFacePoseDetector_pickles/Models/" + fileName,
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

  #  sgd = keras.optimizers.Adam(learning_rate=0.5)

    model.compile(optimizer=sgd,
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    history = model.fit(profileImages, profileDOFlabels,
                        validation_split=0.33,
                        callbacks=callback_list,
                        epochs=1000,
                        batch_size=14,
                        shuffle=True,
                        verbose=1)

    pickle.dump(history.history, open("./ProfileFacePoseDetector_pickles/profileImagesModel_history.pkl", "wb"))

    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

    print("")


testProfile()
