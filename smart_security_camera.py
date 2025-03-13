import cv2
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import face_recognition
import time
import numpy as np

# Email configuration
sender_email = 'specify your mail id'
receiver_email = 'specify receiver mail id '
password = 'your password'
smtp_server = 'smtp.gmail.com'
smtp_port = 587

# Load known face encodings and names
known_face_encodings = []
known_face_names = []

# Load images of known faces and encode them
for file_name in os.listdir('known_faces'):
    image = face_recognition.load_image_file(f'known_faces/{file_name}')
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(os.path.splitext(file_name)[0])  # Use the file name as the name

# Initialize the camera
cap = cv2.VideoCapture(2)
time.sleep(1)  # Give time for the camera to start

# Initialize variables for motion detection
ret, initial_frame = cap.read()
initial_frame = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
initial_frame = cv2.GaussianBlur(initial_frame, (21, 21), 0)
last_sent_time = time.time()  # Track the last time an email was sent
email_interval = 2  # Minimum seconds between emails
frame_refresh_interval = 2  # Seconds between initial_frame updates
last_frame_update = time.time()

# Function to send email with the captured image
def send_email(image_path):
    try:
        print("Preparing to send email...")
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = 'Motion Detected! Unknown Face Detected'
        
        # Attach the image to the email
        with open(image_path, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(image_path)}')
            msg.attach(part)

        # Connect to the SMTP server and send the email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, password)
            server.send_message(msg)
            print("Email sent successfully!")

    except Exception as e:
        print(f"Failed to send email: {e}")

# Main loop
print("Starting the smart security camera with face recognition...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert the frame to grayscale and apply Gaussian blur for motion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # Update the reference frame every `frame_refresh_interval` seconds
    if time.time() - last_frame_update > frame_refresh_interval:
        initial_frame = gray.copy()
        last_frame_update = time.time()

    # Compute the absolute difference between the initial frame and the current frame
    frame_delta = cv2.absdiff(initial_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Count the non-zero pixels to measure motion intensity
    motion_area = cv2.countNonZero(thresh)
    motion_detected = motion_area > 7000  # Adjust this threshold based on testing

    unknown_face_detected = False  # Flag to track if unknown face is detected

    if motion_detected:
        # Detect faces in the frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            # Compare face with known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if True not in matches:
                unknown_face_detected = True  # Mark that we have an unknown face
            else:
                print("Known face detected, no alert sent.")
        
        # If an unknown face is detected and enough time has passed since the last email
        if unknown_face_detected and (time.time() - last_sent_time > email_interval):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            image_path = f'captured_image_{timestamp}.jpg'
            cv2.imwrite(image_path, frame)  # Save the captured image
            print("Unknown face detected! Capturing image...")
            
            # Send the email with the captured image
            send_email(image_path)
            last_sent_time = time.time()  # Update last sent time

    # Display the frame and threshold for debugging
    cv2.imshow('Smart Security Camera', frame)
    cv2.imshow('Threshold', thresh)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
