import face_recognition
import cv2
import os
import pandas as pd
from datetime import datetime


# Step 1: Load Known Faces and Encode Them
def load_known_faces(known_faces_dir):
    known_encodings = []
    known_names = []

    for filename in os.listdir(known_faces_dir):
        if filename.endswith(('.jpg', '.png')):
            filepath = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)

            if encodings:  # Ensure a face was detected in the image
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(filename)[0])

    return known_encodings, known_names


# Step 2: Initialize Attendance Log
def initialize_log(log_path):
    if not os.path.exists(log_path):
        columns = ['Name', 'Date', 'Time']
        df = pd.DataFrame(columns=columns)
        df.to_csv(log_path, index=False)


# Step 3: Mark Attendance
def mark_attendance(name, log_path):
    now = datetime.now()
    date = now.strftime('%Y-%m-%d')
    time = now.strftime('%H:%M:%S')

    # Load existing log
    df = pd.read_csv(log_path)

    # Check if the person is already logged for today
    if not ((df['Name'] == name) & (df['Date'] == date)).any():
        new_entry = {'Name': name, 'Date': date, 'Time': time}
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        df.to_csv(log_path, index=False)
        print(f"Attendance marked for {name} at {time}.")
    else:
        print(f"{name} is already marked present for today.")


# Step 4: Real-Time Face Recognition
def recognize_faces(video_source, known_encodings, known_names, log_path):
    cap = cv2.VideoCapture(video_source)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from video. Exiting.")
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces and compute encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            # If a match is found, get the name
            if True in matches:
                match_index = matches.index(True)
                name = known_names[match_index]
                mark_attendance(name, log_path)

            # Draw rectangle and label on the frame
            top, right, bottom, left = [v * 4 for v in face_location]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Attendance System', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Main Program
if __name__ == "__main__":
    KNOWN_FACES_DIR = 'known_faces'
    ATTENDANCE_LOG = 'logs/attendance.csv'
    VIDEO_SOURCE = 0  # Use webcam as video source

    # Step 1: Load known faces
    known_encodings, known_names = load_known_faces(KNOWN_FACES_DIR)

    # Step 2: Initialize log file
    initialize_log(ATTENDANCE_LOG)

    # Step 3: Start real-time face recognition
    recognize_faces(VIDEO_SOURCE, known_encodings, known_names, ATTENDANCE_LOG)