# SkinkaAI
### Project: Face and T-Zone Detection with Skin Tone Analysis

This project integrates real-time face detection, T-Zone identification, and skin tone analysis using Python libraries like `OpenCV`, `dlib`, and `scikit-learn`. The purpose of the project is to detect faces, extract the T-Zone (forehead, nose, and upper lip area), and analyze dominant skin tones within this region for further applications like dermatological assessments or beauty industry solutions.

Key Features:

1. Face Detection:
   - Utilizes `dlib`'s pre-trained frontal face detector for accurate and efficient face localization in both images and real-time video streams.

2. T-Zone Identification:
   - Dynamically defines the T-Zone region (central face area) using proportions relative to the detected face's dimensions.
   - Highlights the T-Zone area with bounding boxes in live video feeds for visualization.

3. Skin Tone Analysis:
   - Applies **K-Means clustering** to identify dominant skin tones in the T-Zone region.
   - Smoothens the color detection results using a rolling buffer for consistent and stable output.
   - Displays dominant skin colors in real time along with their HEX codes for easy interpretation.

4. Preprocessing Techniques:
   - Converts frames to grayscale for face detection.
   - Applies noise reduction techniques like bilateral filtering to improve skin tone detection accuracy.

5. Real-Time Visualization:
   - Displays real-time video feed with face bounding boxes and T-Zone highlights.
   - Shows a separate window for the T-Zone area and overlays dominant skin colors on the main frame.



This project demonstrates the synergy between computer vision and machine learning for analyzing facial features and regions, offering a foundation for further advancements in healthcare, beauty, and entertainment industries.
