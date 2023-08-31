import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import legacy
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import Recall,Precision
from tensorflow.keras.callbacks import EarlyStopping
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# command line argument
#-------------------
ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="train/display")
mode = ap.parse_args().mode
#-------------------

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes
#old model
faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"
faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']
#-------------------------------------------------------------------------
# plots accuracy, loss, precision and recall curves
def plot_model_history(model_history):
    """
    Plot Accuracy,Loss,Precision and Recall curves given the model_history
    """
    fig, axs = plt.subplots(2,2,figsize=(15,5))#make it in 2 line
    # summarize history for accuracy
    axs[0][0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'])
    axs[0][0].plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'])
    axs[0][0].set_title('Model Accuracy')
    axs[0][0].set_ylabel('Accuracy')
    axs[0][0].set_xlabel('Epoch')
    axs[0][0].set_xticks(np.arange(1, len(model_history.history['accuracy']) + 1, len(model_history.history['accuracy']) / 10))  # ,10 or len(model_history.history['accuracy'])/10
    axs[0][0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[0][1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[0][1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[0][1].set_title('Model Loss')
    axs[0][1].set_ylabel('Loss')
    axs[0][1].set_xlabel('Epoch')
    axs[0][1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1, len(model_history.history['loss']) / 10))  # ,10 or len(model_history.history['loss'])/10
    axs[0][1].legend(['train', 'val'], loc='best')
    # summarize history for recall
    axs[1][0].plot(range(1, len(model_history.history['recall']) + 1), model_history.history['recall'])
    axs[1][0].plot(range(1, len(model_history.history['val_recall']) + 1), model_history.history['val_recall'])
    axs[1][0].set_title('Model recall')
    axs[1][0].set_ylabel('recall')
    axs[1][0].set_xlabel('Epoch')
    axs[1][0].set_xticks(np.arange(1, len(model_history.history['recall']) + 1, len(model_history.history['recall']) / 10))  # ,10 or len(model_history.history['recall'])/10
    axs[1][0].legend(['train', 'val'], loc='best')
    # summarize history for precision
    axs[1][1].plot(range(1, len(model_history.history['precision']) + 1), model_history.history['precision'])
    axs[1][1].plot(range(1, len(model_history.history['val_precision']) + 1), model_history.history['val_precision'])
    axs[1][1].set_title('Model precision')
    axs[1][1].set_ylabel('precision')
    axs[1][1].set_xlabel('Epoch')
    axs[1][1].set_xticks(np.arange(1, len(model_history.history['precision']) + 1, len(model_history.history['precision']) / 10))  # ,10 or len(model_history.history['precision'])/10
    axs[1][1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    fig.tight_layout(pad=1.0)
    plt.show()

# Define data generators
#-------------------
train_dir = 'data/train'
val_dir = 'data/test'
num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 100
#-------------------

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))


# If you want to train the same model or try other models, go for this
if mode == "train":
    model.compile(loss='categorical_crossentropy',optimizer=legacy.Adam(learning_rate=0.0001, decay=1e-6),metrics=['accuracy',Precision(),Recall()])
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    model_info = model.fit(
            train_generator,
            steps_per_epoch=num_train // batch_size,
            epochs=num_epoch,
            validation_data=validation_generator,
            validation_steps=num_val // batch_size,
            callbacks = [early_stopping])#verbose=0 if want silenc =1 animate like " [=====] " =2 is "epoch 1/10"
    plot_model_history(model_info)
    model.save_weights('best.h5')

elif mode == "display":
    video = cv2.VideoCapture(0) # 0 for pc camera 1 for cam camera
    padding = 20
    model.load_weights('best.h5')
    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)
    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    while cv2.waitKey(1)<0:
        hasFrame,frame=video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not hasFrame:
            cv2.waitKey()
            break

        resultImg,faceBoxes=highlightFace(faceNet,frame)
        if not faceBoxes:
            print("No face detected")

        for faceBox in faceBoxes:
            face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

            blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds=genderNet.forward()
            gender=genderList[genderPreds[0].argmax()]
            print(f'Gender: {gender}')

            ageNet.setInput(blob)
            agePreds=ageNet.forward()
            age=ageList[agePreds[0].argmax()]
            print(f'Age: {age[1:-1]} years')

            roi_gray = gray[faceBox[1]:faceBox[1] + faceBox[3], faceBox[0]:faceBox[0] + faceBox[2]]
            if roi_gray.shape[0] > 0 and roi_gray.shape[1] > 0:  #new Check if the face region is valid
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img, verbose=0)#verbose=0 to hide the animate [=========]
            print(f'Emotion: {emotion_dict[int(np.argmax(prediction))]}')

            cv2.putText(resultImg, f'{gender}, {age}, {emotion_dict[int(np.argmax(prediction))]}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
            cv2.imshow("Detecting gender, age and emotion",resultImg)
