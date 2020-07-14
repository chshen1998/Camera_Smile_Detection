import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('CustomSmileDetector.h5')
def is_smiling(img):
    pred = model.predict(img)
    return pred[0][0]

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

isWaiting = False
takePhoto = False

while cap.isOpened():
    _, img = cap.read()

    if (takePhoto):
        cv2.imwrite('Screenshots/1.png', img)
        takePhoto = False

    if isWaiting:
        faces = face_cascade.detectMultiScale(img, 1.2, 4)
        num_faces = 0
        num_smiles = 0

        for (x, y, w, h) in faces:
            num_faces += 1
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (64,64))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_array = np.array(face)
            face_array = np.expand_dims(face_array, axis=0)
            face_array = np.expand_dims(face_array, axis=-1)
            if (is_smiling(face_array) >= 0.5):
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(img, 'SMILING', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
                num_smiles += 1
                if (num_faces == num_smiles):
                    takePhoto = True
                    isWaiting = False
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(img, 'NOT SMILING', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Camera', img)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        isWaiting = True
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()

