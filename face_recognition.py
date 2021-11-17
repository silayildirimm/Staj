import cv2
face_cascade = cv2.CascadeClassifier('C:\\Users\\SILA\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier('C:\\Users\\SILA\\opencv\\sources\\data\\haarcascades\\haarcascade_eye.xml')

video = cv2.VideoCapture(0)
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize= (30,30), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes=eye_cascade.detectMultiScale(roi_gray,1.1,18,minSize=(50,50))
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (255,0,0),2)
            
    cv2.imshow("video",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()