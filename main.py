import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('emotion_model.h5')
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    cv2.imwrite("img.png", img)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("gray.png", gray)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cv2.imwrite("roi_gray.png", roi_gray)
        small = cv2.resize(roi_gray, (48,48))
        fitted = np.array([small])
        ypred = model.predict(fitted)
        emot_ind = ypred[0].argsort()[-2:][::-1]
        cv2.putText(img, emotion_dict[emot_ind[0]], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,255), 1, 1 )
        cv2.putText(img, emotion_dict[emot_ind[1]], (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,255), 1, 1 )

    cv2.imshow('Image', img)
    cv2.imwrite("img_final.png", img)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cap.destroyAllWindows()