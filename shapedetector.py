"""
shape_detector.py
Detect basic shapes (triangle, square, rectangle, pentagon, hexagon, circle)
in an image or webcam stream using OpenCV.

Usage:
    python shape_detector.py --image path/to/image.jpg
or
    python shape_detector.py --webcam 0

Dependencies:
    pip install opencv-python numpy
"""

import cv2
import numpy as np
import argparse

def get_shape_name(cnt):
    # Approximate the contour to reduce number of vertices
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    v = len(approx)

    if v == 3:
        return "Triangle"
    elif v == 4:
        # Distinguish square vs rectangle by aspect ratio of bounding box
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        return "Square" if 0.95 <= ar <= 1.05 else "Rectangle"
    elif v == 5:
        return "Pentagon"
    elif v == 6:
        return "Hexagon"
    else:
        # Check for circle-like by comparing contour area to area of min enclosing circle
        area = cv2.contourArea(cnt)
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        circle_area = np.pi * (radius ** 2)
        if radius > 5 and area / circle_area > 0.75:
            return "Circle"
        else:
            return f"{v}-sided"

def detect_and_annotate(image, draw_contours=True, min_area=200):
    """
    Detect shapes in a BGR image and annotate it.
    Returns annotated image and list of detections (name, centroid, contour).
    """
    annotated = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    # Use adaptive threshold or Canny depending on the image. Canny is versatile:
    edged = cv2.Canny(blur, 50, 150)
    # Dilate + erode to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        shape_name = get_shape_name(cnt)

        # Draw contour and label
        if draw_contours:
            cv2.drawContours(annotated, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(annotated, (cx, cy), 3, (0,0,255), -1)

        # Put text slightly above centroid
        cv2.putText(annotated, shape_name, (cx - 40, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        detections.append((shape_name, (cx, cy), cnt))

    return annotated, detections

def run_on_image(path):
    img = cv2.imread(path)
    if img is None:
        print("ERROR: Could not read image:", path)
        return

    annotated, det = detect_and_annotate(img)
    print("Detections:")
    for name, (cx, cy), _ in det:
        print(f" - {name} at ({cx},{cy})")
    cv2.imshow("Shapes", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def run_on_webcam(cam_index=0):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("ERROR: Could not open webcam:", cam_index)
        return

    # Set higher resolution for better view
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Shape Detector (webcam)", cv2.WINDOW_NORMAL)  # Resizable window

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, _ = detect_and_annotate(frame)

        # Resize to a big size but keep title bar
        annotated = cv2.resize(annotated, (1280, 720))  # Change size if needed

        cv2.imshow("Shape Detector (webcam)", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple shape detector using OpenCV")
    parser.add_argument("--image", help="Path to input image")
    parser.add_argument("--webcam", type=int, help="Open webcam by index (e.g. 0)")
    args = parser.parse_args()

    if args.image:
        run_on_image(args.image)
    elif args.webcam is not None:
        run_on_webcam(args.webcam)
    else:
        print("Provide --image <path> or --webcam <index>. Example: python shape_detector.py --image shapes.jpg")
