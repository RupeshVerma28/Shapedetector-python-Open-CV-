# 🛑 Object Shape Detector (Python + OpenCV)

A simple **Shape Detection** application built with **Python** and **OpenCV** that detects basic shapes like **Triangle, Square, Rectangle, Pentagon, Hexagon, Circle**, and more.  
It works with **both images** and **live webcam feed**.

---

## 📌 Features
- Detects and labels:
  - Triangle
  - Square
  - Rectangle
  - Pentagon
  - Hexagon
  - Circle
  - Any polygonal shape (shows number of sides)
- Works with:
  - Image files
  - Live webcam
- Highlights contours and centroids
- Resizable window with close button
- Adjustable sensitivity and minimum area filter

---

## 📂 Project Structure
```
ShapeDetector/
│
├── shape_detector.py   # Main Python script
├── shapes.jpg          # Sample test image (optional)
└── README.md           # Project documentation
```

---

## 🔧 Requirements
- Python 3.7+
- OpenCV
- NumPy

Install dependencies:
```bash
pip install opencv-python numpy
```

---

## 🚀 Usage

### 1️⃣ Run with an Image
```bash
python shape_detector.py --image shapes.jpg
```

### 2️⃣ Run with Webcam
```bash
python shape_detector.py --webcam 0
```
> Press **`q`** to quit the webcam window.

---

## ⚙️ Configuration

Inside `shape_detector.py`, you can adjust:
- **Resolution**:
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```
- **Minimum Shape Area**:
```python
min_area=200  # Increase to ignore small noise
```
- **Polygon Approximation Precision**:
```python
approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
```
Lower the `0.04` for more precise shape detection.

---

## 📸 Example Output

**Image Input:**  
shapes.jpg

**Detection Output:**  
(Insert detection screenshot here)

---

## 📝 Notes
- Ensure your environment has access to the webcam for live detection.
- Good lighting and clear shapes improve accuracy.
- For complex/overlapping shapes, consider preprocessing with color filtering.

---

## 📜 License
This project is **open-source** under the MIT License.

---

## 👨‍💻 Author
**Rupesh Verma**  
💻 Full Stack Developer | UI/UX Designer | Cybersecurity Enthusiast  
📧 Email: errupesh28@gmail.com
