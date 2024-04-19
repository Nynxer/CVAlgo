import cv2 as cv
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Define list of people and training directory
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling', 'Nikhil Kamath']
DIR = r'Faces\train'

haar_cascade = cv.CascadeClassifier('haar_face.xml')
if haar_cascade.empty():
    print("Error loading the cascade classifier.")
    sys.exit()

features, labels = [], []

def process_images(directory):
    for person in people:
        path = os.path.join(directory, person)
        label = people.index(person)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv.imread(img_path)
            if img is None:
                print(f"Warning: Unable to load image {img_path}")
                continue
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                faces_roi_resized = cv.resize(faces_roi, (200, 200))
                features.append(faces_roi_resized)
                labels.append(label)

process_images(DIR)


features = np.array(features, dtype='uint8')
labels = np.array(labels, dtype='int32')

# Data Augmentation: Create flipped images
augmented_features = []
augmented_labels = []
for feature, label in zip(features, labels):
    flipped_feature = cv.flip(feature, 1)  # Horizontal flip
    augmented_features.append(flipped_feature)
    augmented_labels.append(label)

total_features = np.concatenate((features, augmented_features), axis=0)
total_labels = np.concatenate((labels, augmented_labels), axis=0)

models = {
    "LBPH": cv.face.LBPHFaceRecognizer_create(),
    "Eigenface": cv.face.EigenFaceRecognizer_create(),
    "Fisherface": cv.face.FisherFaceRecognizer_create()
}
for name, model in models.items():
    model.train(total_features, total_labels)
    model.save(f'face_trained_{name}.yml')

print('All models trained and data saved.')