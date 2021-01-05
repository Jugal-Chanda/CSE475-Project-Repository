from keras.models import load_model
import cv2
import numpy as np

model = load_model('model.h5')

cv2.ocl.setUseOpenCL(False)

emotions = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1020)
cap.set(10, 100)

while True:
    success, img = cap.read()
    if not success:
        print("something Error... Check Camera Connection")
        break
    boundingBox = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    num_faces = boundingBox.detectMultiScale(grayImg, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in num_faces:
        cv2.rectangle(img, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        grayImg = grayImg[y:y + h, x:x + w]
        croppedImg = np.expand_dims(np.expand_dims(cv2.resize(grayImg, (48, 48)), -1), 0)
        emotion_prediction = model.predict(croppedImg)
        maxIndex = int(np.argmax(emotion_prediction))
        cv2.putText(img, emotions[maxIndex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Vedio', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
cap.release()
cv2.destroyAllWindows()