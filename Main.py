import cv2

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 20)

        label = "Neutral"
        color = (255, 255, 255)

        if len(smiles) > 0:
            label = "Happy"
            color = (0, 255, 0)
        elif len(eyes) == 0:
            label = "Sad"
            color = (255, 0, 0)
        elif len(eyes) >= 2 and h/w > 1.2:
            label = "Surprised"
            color = (0, 255, 255)
        elif len(eyes) >= 2 and w/h > 1.2:
            label = "Angry"
            color = (0, 0, 255)
        elif len(eyes) == 1:
            label = "Fear"
            color = (255, 255, 0)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Improved Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
