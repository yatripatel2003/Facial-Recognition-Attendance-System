import cv2
import numpy as np
import os
import sqlite3
from datetime import date
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Function to align faces using OpenCV's face detection
def align_face(image, face_rect):
    x, y, w, h = face_rect
    aligned_face = image[y:y+h, x:x+w]
    aligned_face = cv2.resize(aligned_face, (256, 256))  # Resize for consistency
    return aligned_face

# Function to train a face recognition model using a machine learning algorithm
def train_face_recognition(data_dir):
    faces = []
    labels = []

    for root, dirs, files in os.walk(data_dir):
        for label, dir_name in enumerate(dirs):
            for file in os.listdir(os.path.join(root, dir_name)):
                if file.endswith("jpg") or file.endswith("png"):
                    path = os.path.join(root, dir_name, file)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    faces.append(img)
                    labels.append(dir_name)

    # Convert labels to integers
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Train SVM classifier
    svm = SVC(kernel='linear', probability=True)
    svm.fit(np.array(faces), labels)

    return svm, label_encoder.classes_

# Function to recognize faces with the trained face recognition model
def recognize_faces_and_mark_attendance(svm, classes, db_file):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        today = date.today()
        table_name = "attendance_" + today.strftime("%Y_%m_%d")
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (id INTEGER PRIMARY KEY, name TEXT)")
        print(f"Table '{table_name}' created successfully.")

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                aligned_face = align_face(gray, (x, y, w, h))
                label = svm.predict(aligned_face.reshape(1, -1))[0]
                confidence = np.max(svm.predict_proba(aligned_face.reshape(1, -1)))
                name = classes[label]
                
                if confidence > 0.5:  # Adjust confidence threshold as needed
                    mark_attendance(cursor, table_name, name)
                    print(f"Attendance marked for {name} (Confidence: {confidence:.2f}).")
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        conn.commit()
        conn.close()
    except Exception as e:
        print("An error occurred:", e)

# Function to mark attendance in the database
def mark_attendance(cursor, table_name, name):
    cursor.execute(f"INSERT INTO {table_name} (name) VALUES (?)", (name,))

if __name__ == "__main__":
    data_dir = "dataset"  # Directory containing labeled face images for training
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    svm, classes = train_face_recognition(data_dir)
    db_file = "attendance.db"  # SQLite database file
    recognize_faces_and_mark_attendance(svm, classes, db_file)
