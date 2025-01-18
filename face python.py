
#HOW DOES COMPUTER DETECT FACES STEP1
import cv2
import dlib
import matplotlib.pyplot as plt
# Load dlib's face detector
detector = dlib.get_frontal_face_detector()
image_path = r"C:\Users\Paru\OneDrive\Pictures\WhatsApp Image 2024-03-24 at 7.12.55 PM.jpeg"
image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = detector(gray)
#SEGMENTATION START
for face in faces:
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.axis("off")  
plt.title("Detected Faces")
plt.show()


#LIVE FEED DETECTION OF FACE STEP2
import cv2
import dlib
detector = dlib.get_frontal_face_detector()

# Start video capture
cap = cv2.VideoCapture(0)#for my cam
while True:
    ret, frame = cap.read() 
    if not ret:
        break

   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = detector(gray)


    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    
    cv2.imshow("Real-Time Face Detection", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#PRESS q TO QUIT
cap.release()
cv2.destroyAllWindows()


#ROI DETECTION-30%,25%,50% STEP3 
import cv2
import dlib
detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)  
while True:
    ret, frame = cap.read()  
    if not ret:
        break

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    faces = detector(gray)

    for face in faces:
        
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        
        t_zone_x1 = x + int(w * 0.2)  
        t_zone_x2 = x + int(w * 0.8)  
        t_zone_y1 = y + int(h * 0.15)  
        t_zone_y2 = y + int(h * 0.60) 
        cv2.rectangle(frame, (t_zone_x1, t_zone_y1), (t_zone_x2, t_zone_y2), (0, 0, 255), 2)

    cv2.imshow("Face and T-Zone Detection", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

#STEP 4 REFINING ROI
import cv2
import dlib
detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)  
while True:
    ret, frame = cap.read()  
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        t_zone_x1 = x + int(w * 0.2)  
        t_zone_x2 = x + int(w * 0.8)  
        t_zone_y1 = y + int(h * 0.15)  
        t_zone_y2 = y + int(h * 0.60)
        t_zone_roi = frame[t_zone_y1:t_zone_y2, t_zone_x1:t_zone_x2]
        t_zone_preprocessed = cv2.GaussianBlur(t_zone_roi, (5, 5), 0)  # Reduce noise
        cv2.imshow("T-Zone ROI", t_zone_preprocessed)
    cv2.imshow("Face and T-Zone Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


#STEP4 ML-UNSUPERVISED LEARNING - KMEANS
import cv2
import dlib
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
from collections import deque
def extract_dominant_colors(image, k=3):
    """
    Extract dominant colors using KMeans clustering.
    k: Number of dominant colors to extract.
    Returns: List of dominant colors (RGB) and their HEX equivalents.
    """
    data = image.reshape((-1, 3))  # Flatten image
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    colors = kmeans.cluster_centers_
    hex_colors = [mcolors.rgb2hex(color / 255) for color in colors]
    return colors, hex_colors
detector = dlib.get_frontal_face_detector()

color_buffer = deque(maxlen=5)  
cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
#refined ROI again(T-ZONE)
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        t_zone_x1 = x + int(w * 0.1)  # 10% from the left of the face
        t_zone_x2 = x + int(w * 0.9)  # 90% from the left of the face
        t_zone_y1 = y + int(h * 0.1)  # 10% from the top of the face
        t_zone_y2 = y + int(h * 0.7)  # 70% from the top of the face
        t_zone_roi = frame[t_zone_y1:t_zone_y2, t_zone_x1:t_zone_x2]
        t_zone_resized = cv2.resize(t_zone_roi, (150, 150))
        t_zone_filtered = cv2.bilateralFilter(t_zone_resized, d=9, sigmaColor=75, sigmaSpace=75)

        dominant_colors, hex_colors = extract_dominant_colors(t_zone_filtered, k=3)
        color_buffer.append(dominant_colors)
        avg_colors = np.mean(color_buffer, axis=0)
        hex_avg_colors = [mcolors.rgb2hex(color / 255) for color in avg_colors] #HEX COLOURS

        for i, color in enumerate(avg_colors):
            color_bgr = tuple(int(c) for c in color) 
            cv2.rectangle(frame, (10, 50 * i + 10), (60, 50 * i + 40), color_bgr, -1)
            cv2.putText(frame, hex_avg_colors[i], (70, 50 * i + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imshow("T-Zone ROI", t_zone_roi)

    cv2.imshow("Face and T-Zone Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
quit()