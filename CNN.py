import cv2
import numpy as np
import os
import sqlite3
from datetime import date
import tensorflow as tf

# Function to train the CNN face recognition model
def train_cnn_face_recognition(data_dir, input_shape, num_classes, batch_size=32, epochs=20):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(input_shape[0], input_shape[1]),
        batch_size=batch_size,
        class_mode='categorical')

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.04),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    model.fit(train_generator, epochs=epochs)

    return model, train_generator.class_indices

# Function to recognize faces using CNN model and mark attendance
def recognize_faces_cnn(model, label_to_name, db_file):
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
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi_color = frame[y:y+h, x:x+w]
                roi_color = cv2.resize(roi_color, (100, 100))
                roi_color = np.expand_dims(roi_color, axis=0)  # Add batch dimension
                roi_color = roi_color / 255.0  # Normalize

                predictions = model.predict(roi_color)
                predicted_class = np.argmax(predictions)

                confidence_threshold = 0.8

                if predictions[0][predicted_class] > confidence_threshold:
                    name = label_to_name.get(predicted_class, "Unknown")
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

# Function to train the LBPH face recognition model
def train_lbph_face_recognition(data_dir):
    recognizer = cv2.facu e.LBPHFaceRecognizer_create()
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

# Function to recognize faces using LBPH model and mark attendance
def recognize_faces_lbph(recognizer, label_to_name, db_file):
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
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                label, confidence = recognizer.predict(roi_gray)

                confidence_threshold = 70

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
    data_dir_lbph = "dataSet"  # Directory containing labeled face images for LBPH training
    lbph_recognizer, lbph_label_to_name = train_lbph_face_recognition(data_dir_lbph)

    data_dir_cnn = "dataSet"  # Directory containing labeled face images for CNN training
    input_shape = (100, 100, 3)  # Input shape for CNN model
    num_classes = len(os.listdir(data_dir_cnn))  # Number of classes (persons)
    cnn_model, cnn_label_to_name = train_cnn_face_recognition(data_dir_cnn, input_shape, num_classes)

    db_file = "attendance.db"  # SQLite database file

    print("Choose Recognition Mode:")
    print("1. LBPH")
    print("2. CNN")
    recognition_mode = input("Enter mode (1 or 2): ")

    if recognition_mode == "1":
        recognize_faces_lbph(lbph_recognizer, lbph_label_to_name, db_file)
    elif recognition_mode == "2":
        recognize_faces_cnn(cnn_model, cnn_label_to_name, db_file)
    else:
        print("Invalid mode selected.")
