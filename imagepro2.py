import cv2
import numpy as np
import os
import sqlite3
from datetime import date

# Function to train the face recognition model
def train_face_recognition(data_dir):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    faces = []
    labels = []
    label_to_name = {}

    for root, dirs, files in os.walk(data_dir):
        for label, dir_name in enumerate(dirs):
            label_to_name[label] = dir_name
            for file in os.listdir(os.path.join(root, dir_name)):
                if file.endswith("jpg") or file.endswith("png"):
                    path = os.path.join(root, dir_name, file)
                    face_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    faces.append(face_img)
                    labels.append(label)

    recognizer.train(faces, np.array(labels))

    return recognizer, label_to_name

# Function to recognize faces in webcam feed and mark attendance
def recognize_faces_and_mark_attendance(recognizer, label_to_name, db_file):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        today = date.today()
        table_name = "attendance_" + today.strftime("%Y_%m_%d")

        # Create a table for today's attendance if it doesn't exist
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (id INTEGER PRIMARY KEY, name TEXT)")
        print(f"Table '{table_name}' created successfully.")

        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Face alignment
                face_roi = gray[y:y+h, x:x+w]
                aligned_face = cv2.resize(face_roi, (100, 100))

                label, confidence = recognizer.predict(aligned_face)

                # Adjust confidence threshold dynamically based on recognition scenario
                confidence_threshold = 50
                if len(faces) > 1:
                    confidence_threshold = 60  # Increase threshold if multiple faces detected
                elif len(faces) == 0:
                    continue  # Skip recognition if no face detected

                if confidence < confidence_threshold:
                    name = label_to_name.get(label, "Unknown")
                    mark_attendance(cursor, table_name, name)
                    print(f"Attendance marked for {name}.")
                else:
                    name = "Unknown"

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

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
    recognizer, label_to_name = train_face_recognition(data_dir)
    db_file = "attendance.db"  # SQLite database file
    recognize_faces_and_mark_attendance(recognizer, label_to_name, db_file)
