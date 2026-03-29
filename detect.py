import cv2 as cv
import os
import time
print("Singer's Face Recognizer")
haar_cascade=cv.CascadeClassifier("haar_face.xml")
people=["Taylor Swift","jungkook","justin bieber","Ariana Grande"]
face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")
img_path=input("Enter the image path: ").replace('"','')
img=cv.imread(img_path)
if img is None:
    print("Image not found. Check the path.")
    exit()
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("Person",gray)
faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
for (x,y,w,h) in faces_rect:
                faces_roi=gray[y:y+h,x:x+w]
                label,confidence=face_recognizer.predict(faces_roi)
                print(f"Label={people[label]} with a confidence of {confidence}")
                cv.putText(img,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),
                           thickness=2)
                cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
# Create output folder if it doesn't exist
os.makedirs("output", exist_ok=True)
# Create unique filename using time
filename = f"result_{int(time.time())}.jpg"
# Create full path
output_path = os.path.join("output", filename)
# Save image
cv.imwrite(output_path, img)
print(f"Image saved at: {output_path}")
cv.imshow("Detected Face",img)
cv.waitKey(0)
cv.destroyAllWindows()