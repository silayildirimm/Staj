import cv2
face_cascade = cv2.CascadeClassifier('C:\\Users\\SILA\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier('C:\\Users\\SILA\\opencv\\sources\\data\\haarcascades\\haarcascade_eye.xml')

img = cv2.imread("C:\\Users\\SILA\\.spyder-py3\\resim.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize = (30,30), flags = cv2.CASCADE_SCALE_IMAGE)

for(x,y,w,h) in faces:
    img= cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 18)
    
    for(x1,y1,w1,h1) in eyes:
        cv2.rectangle(roi_color, (x1,y1),(x1+w1, y1+h1), (255,0,0),2)
        
cv2.imshow('img',img)
cv2.imwrite("output_detected.jpg",img)
cv2.waitKet(0)
cv2.destroyAllWindows()