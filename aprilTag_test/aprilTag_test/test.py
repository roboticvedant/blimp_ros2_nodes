import cv2
import apriltag
import numpy as np
import yaml
import os

# Function to load camera calibration from a YAML file
def load_camera_calibration(calib_file_path):
    with open(calib_file_path, "r") as file:
        calib_data = yaml.safe_load(file)
    
    camera_matrix = np.array(calib_data["camera_matrix"]["data"]).reshape(3, 3)
    dist_coeffs = np.array(calib_data["distortion_coefficients"]["data"])
    return camera_matrix, dist_coeffs

# Path to the calibration file
calib_file_path = os.path.expanduser("~/Desktop/ost.yaml")

# Load calibration parameters
try:
    camera_matrix, dist_coeffs = load_camera_calibration(calib_file_path)
    print("Camera calibration loaded successfully.")
except Exception as e:
    print(f"Error loading calibration file: {e}")
    exit()

# Tag size in meters
tag_size = 0.100  # 100 mm

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize the AprilTag detector
options = apriltag.DetectorOptions(families="tag25h9")
detector = apriltag.Detector(options)

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags
    detections = detector.detect(gray)

    for detection in detections:
        # Estimate the pose of the tag
        object_points = np.array([
            [-tag_size / 2, -tag_size / 2, 0],  # Bottom-left corner
            [ tag_size / 2, -tag_size / 2, 0],  # Bottom-right corner
            [ tag_size / 2,  tag_size / 2, 0],  # Top-right corner
            [-tag_size / 2,  tag_size / 2, 0]   # Top-left corner
        ])

        image_points = np.array(detection.corners)

        retval, rvec, tvec = cv2.solvePnP(
            objectPoints=object_points,
            imagePoints=image_points,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs
        )

        if retval:
            # Translation vector (x, y, z)
            print(f"Translation Vector (x, y, z): {tvec.ravel()}")

            # Rotation vector -> Rotation matrix
            rmat, _ = cv2.Rodrigues(rvec)
            print(f"Rotation Matrix:\n{rmat}")

            # Display the tag ID
            cv2.putText(frame, f"ID: {detection.tag_id}", 
                        (int(detection.center[0]), int(detection.center[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw the detection outline
            for i in range(4):
                start = tuple(map(int, detection.corners[i]))
                end = tuple(map(int, detection.corners[(i + 1) % 4]))
                cv2.line(frame, start, end, (0, 255, 0), 2)

    # Show the video feed with tag detections
    cv2.imshow("AprilTag Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
