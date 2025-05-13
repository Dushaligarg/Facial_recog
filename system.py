import time
import datetime
import csv
import face_recognition_models
import dlib
import face_recognition
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog
from tkinter import messagebox
import cv2


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def add_employee_btn():
    def add_employee():
        cap = cv2.VideoCapture(0)
        messagebox.showinfo("Instructions", """To stop adding Employee or quit adding  employees 
press 'q' on your keyboard. & To click the picture of employee press 'c' on your keyboard.""")

        # Create a window for preview
        cv2.namedWindow("Employee Preview")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture frame")
                break

            # Display the preview frame
            cv2.imshow("Employee Preview", frame)

            # Check for key press (q to quit, c to capture)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Capture the frame when 'c' is pressed
                name = simpledialog.askstring("Name", "Enter employee name:")
                if name is None:
                    continue  # Skip if name is not entered
                image_name = name + ".jpg"
                image_path = os.path.join("./FR/Images", image_name)
                cv2.imwrite(image_path, frame)
                print("Image saved as:", image_path)
                break

        cap.release()
        cv2.destroyWindow("Employee Preview")

    add_employee()

def mark_attendance():
    def write_to_csv(file_path, data):
        with open(file_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            
            if file.tell() == 0:
                writer.writeheader()
            
            writer.writerow(data)

    def detect_motion(prev_frame, curr_frame):
    # Convert frames to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Calculate absolute difference between frames
        frame_diff = cv2.absdiff(prev_gray, curr_gray)

        # Threshold the difference
        _, frame_diff_thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

        # Count non-zero pixels in the thresholded difference
        motion_pixels = cv2.countNonZero(frame_diff_thresh)

        return motion_pixels

# Function to detect facial landmarks
    def detect_landmarks(frame):
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector(gray)

        # If no face detected, return False
        if len(faces) == 0:
            return False

        # Loop over detected faces
        for face in faces:
            # Detect facial landmarks
            landmarks = predictor(gray, face)

            # Convert landmarks to numpy array
            landmarks = [[p.x, p.y] for p in landmarks.parts()]

        return landmarks

    # Function to determine if face is real or fake
    def is_real_face(frame):
        # Detect motion
        prev_frame = frame.copy()
        curr_frame = frame.copy()
        motion_pixels = detect_motion(prev_frame, curr_frame)

        # If motion detected, it's likely a real face
        if motion_pixels > 500:
            return True

        # If no motion detected, check for facial landmarks
        landmarks = detect_landmarks(frame)
        if landmarks:
            return True  # Real face
        else:
            return False  # Fake face (likely a picture)
    

    def main():
        now_time = time.localtime()
        current_time = time.strftime("%H:%M:%S", now_time)
        date= datetime.date.today()
        timestamp = str(date)+" "+str(current_time)

        folder_path = "./FR/Images" 
        known_face_encodings = [] 
        known_face_names = [] 
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg'):
                image = face_recognition.load_image_file(os.path.join(folder_path, filename))
                face_locations = face_recognition.face_locations(image)
                
                if face_locations:
                    face_encoding = face_recognition.face_encodings(image, known_face_locations=face_locations)[0]
                    name = os.path.splitext(filename)[0]
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(name)
                    print("Encoded:", name)
                else:
                    print("No face found in:", filename)

        video_capture = cv2.VideoCapture(0)

        while True:
            now_time = time.localtime()
            current_time = time.strftime("%H:%M:%S", now_time)
            date= datetime.date.today()
            timestamp = str(date)+" "+str(current_time)


            data = []
            ret, frame = video_capture.read()

            if not ret:
                break

            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            # faces = detector(frame)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Not recognized"

                face_region = frame[top:top+bottom, left:left+right] ## new one

                # for face in faces:
                #     x, y, w, h = face.left(), face.top(), face.width(), face.height()
                #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                #     face_region = frame[y:y+h, x:x+w]

                #     # Check if face is real or fake
                #     if is_real_face(face_region):
                #         cv2.putText(frame, "Real", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                #     else:
                #         cv2.putText(frame, "Fake", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if is_real_face(face_region):
                    cv2.putText(frame, "Real", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Fake", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
                landmarks = detect_landmarks(frame)
                if landmarks:
                    data = {'Time':timestamp, 'Name':name}
                else:
                    pass
                data = {'Time':timestamp, 'Name':name}

                csv_file_path = "./FR/Attendence/attendence.csv"  ##
                # df = pd.DataFrame(data, index=[0]) ##
                write_to_csv(csv_file_path, data=data) ##


            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()
    main()

def check_attendence():
    file_path="C:\\Users\\bb\\FR\\Attendence\\attendence.csv"
    os.startfile(file_path)

def exit_program():
    exit()

def about_team():
    messagebox.showinfo("Team Eagle Details", """
Developer : Team OUTSOURCER & Vaishnav Arora 
Jr. Developer : Manav Sharma
Documentation : Dushali Garg
Lirature : Ridam Bhatia
""")

def show_help_box():
    messagebox.showinfo("Help", """
Add Employee -> Add a new Employee in your Database
Mark Attendence -> Mark attendence of exsisting Employees
Check Attendence -> Check the attendance of all employees or any specific employee
Exit -> Exit the Attendence System
""")

root = tk.Tk()
root.title("Attendence System by Team Eagle")

button_style = {
    "bg": "#4CAF50",
    "width": 20, 
    "height": 2,
    "font": ("Arial", 12)
}

canvas = tk.Canvas(root, width=50, height=50)
canvas.pack(side="right", anchor="se", padx=30, pady=30)  # Align to the bottom-right corner and add some padding

button_center = 25  # Adjusted coordinates to place it in the corner
button = canvas.create_rectangle(0, 0, 50, 50, outline="black", fill="blue")
canvas.create_text(button_center, button_center, text="?", font=("Arial", 15), fill="white")

canvas.tag_bind(button, "<Button-1>", lambda event: show_help_box())


add_employee_button = tk.Button(root, text="Add Employee", command=add_employee_btn, **button_style)
add_employee_button.pack(padx=20,pady=10)
# add_employee_button.pack(side = "left", padx=80,pady=30)


mark_attendance_button = tk.Button(root, text="Mark Attendance", command=mark_attendance, **button_style)
mark_attendance_button.pack(padx=20,pady=10)

check_attendance_button = tk.Button(root, text="Check Attendance", command=check_attendence, **button_style)
check_attendance_button.pack(padx=20,pady=10)

exit_button = tk.Button(root, text="Exit", command=exit_program, **button_style)
exit_button.pack(padx=20,pady=10)

about_button = tk.Button(root, text="About Team Eagle", command=about_team, **button_style)
about_button.pack(padx=20,pady=10)

root.mainloop()
