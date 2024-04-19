import cv2 as cv
import sys

# Load the classifier for face detection
haar_cascade = cv.CascadeClassifier('haar_face.xml')
if haar_cascade.empty():
    print("Error loading the cascade classifier.")
    sys.exit()

# Names corresponding to the labels
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling', 'Nikhil Kamath']
face_count = {name: 0 for name in people}  
def get_face_recognizer(choice):
    if choice == 1:
        recognizer = cv.face.EigenFaceRecognizer_create()
        recognizer.read('face_trained_Eigenface.yml')
        print("Using Eigenfaces algorithm.")
    elif choice == 2:
        recognizer = cv.face.FisherFaceRecognizer_create()
        recognizer.read('face_trained_Fisherface.yml')
        print("Using Fisherfaces algorithm.")
    elif choice == 3:
        recognizer = cv.face.LBPHFaceRecognizer_create()
        recognizer.read('face_trained_LBPH.yml')
        print("Using LBPH algorithm.")
    else:
        raise ValueError("Invalid choice. Please select a valid algorithm.")
    return recognizer

def convert_to_pseudo_confidence(distance, base=10000):
    if distance <= 0:
        return 100  # Max confidence
    else:
        pseudo_confidence = max(0, 100 - (distance / base * 100))
        return pseudo_confidence

# Display algorithm choices and get user input
print("Select the face recognition algorithm:")
print("1: Eigenfaces")
print("2: Fisherfaces")
print("3: LBPH")
algorithm_choice = int(input("Enter your choice (1-3): "))
face_recognizer = get_face_recognizer(algorithm_choice)

# Initialize webcam
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible.")
    sys.exit()

prev_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)  # Adjusted minNeighbors

    for (x, y, w, h) in faces_rect:
        face_roi = gray[y:y+h, x:x+w]
        resized_face_roi = cv.resize(face_roi, (200, 200)) 
        label, confidence = face_recognizer.predict(resized_face_roi)
        pseudo_confidence_percent = convert_to_pseudo_confidence(confidence)

        if prev_frame is not None:
            frame_delta = cv.absdiff(prev_frame[y:y+h, x:x+w], face_roi)
            motion_score = cv.sumElems(frame_delta)[0] / frame_delta.size

            if motion_score > 10 and pseudo_confidence_percent > 50:  
                detected_person = people[label]
                face_count[detected_person] += 1  
                print(f'Label: {detected_person} with pseudo-confidence: {pseudo_confidence_percent:.2f}%')
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv.putText(frame, f'{detected_person} {pseudo_confidence_percent:.2f}% - Live', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv.putText(frame, 'Potential Spoof Detected', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        prev_frame = gray.copy()

    cv.imshow('Webcam Face Recognition', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv.destroyAllWindows()
print("Total faces detected during the session:")
for name, count in face_count.items():
    print(f"{name}: {count}")
