import cv2
import numpy as np
import os
import speech_recognition as sr
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import threading
import pyttsx3 

# Set the path to your dataset
DB = "E:\DL Projects\Gender Classification\DB"

# Create a list to store images and corresponding labels
images=[]
labels=[]

# Load Haarcascade for face detection
face_cascade=cv2.CascadeClassifier('E:\DL Projects\Gender Classification\haarcascade_frontalface_default.xml')


# Initialize the speech recognition recognizer
recognizer = sr.Recognizer()


# Lock for synchronizing access to shared variables
lock = threading.Lock()

text_speech=pyttsx3.init()

# Loop through the dataset folders
for label in os.listdir(DB):
    label_path=os.path.join(DB, label)
    for image_file in os.listdir(label_path):
        image_path = os.path.join(label_path, image_file)

        # read the image
        img=cv2.imread(image_path)
        gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # detect facesin the image
        faces=face_cascade.detectMultiScale(gray_img,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))

        # If a face is detected, add the face region and label to the dataset
        if len(faces)==1:
            x,y,w,h=faces[0]
            face_resize=gray_img[y:y+h,x:x+w]
            face_resize=cv2.resize(face_resize,(100,100))   # Resize for consistency
            images.append(face_resize)
            labels.append(0 if label=='male' else 1)        # Assign labels (0 for male, 1 for female)

# Convert the lists to NumPy arrays
images = np.array(images)
labels = np.array(labels) 


# Convert the images to flat arrays and normalize
images = images.reshape(images.shape[0], -1)  # Flatten each image
images = images.astype(np.float32) / 255.0    # Normalize pixel values to the range [0, 1]


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Train a k-NN model using scikit-learn
knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors as needed
knn.fit(X_train, y_train)


# Initialize the video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# Variables to track gender and confidence
# locked_gender = None
confidence_threshold = 0.8  # You can adjust this threshold based on your requirements
consecutive_frames = 10  # Number of consecutive frames to lock the gender
current_frames = 0


while True:
    ret,frame=cap.read()

    # Convert the frame to grayscale
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected faces
    for (x,y,w,h) in faces:
        face_resize=gray_frame[y:y + h, x:x + w]
        face_resize=cv2.resize(face_resize,(100,100))    # Resize for consistency

        # Reshape the face_roi for prediction
        test_data = face_resize.reshape(1, -1).astype(np.float32)

        # Use the trained k-NN model for prediction
        result = knn.predict(test_data)
        gender = "Male" if result == 0 else "Female"

        # Display the result on the frame
        cv2.putText(frame, f'Gender: {gender}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
       
       # Print the result to the terminal
        print(f'Gender: {gender}')

       

    # Display the frame
    cv2.imshow("Gender Classification", frame)


    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

 # speak result
text_speech.say(f'The gender is {gender}')
text_speech.runAndWait()
