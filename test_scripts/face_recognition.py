import face_recognition
import numpy as np

img1 = face_recognition.load_image_file("./test_scripts/liu1.jpg")

img1_encoding = face_recognition.face_encodings(img1)[0]

img2 = face_recognition.load_image_file('./test_scripts/liu2.jpg')
img2_encoding = face_recognition.face_encodings(img2)[0]


cosine_sim = np.dot(img1_encoding, img2_encoding) / (np.linalg.norm(img1_encoding)*np.linalg.norm(img2_encoding))

print(cosine_sim)