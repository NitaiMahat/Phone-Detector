# Focus Guard - Complete Architecture Explanation

## 📁 Project Structure

```
Project-Phone-Detector/
├── index.html          # UI structure (3 screens)
├── detection-engine.js # All logic & AI processing
├── monitor-ui.css      # Styling
├── yolov5n.onnx       # AI model file
└── package.json       # Node.js server config
```

---

## 🎯 **How Everything Works Together**

### **1. INITIALIZATION FLOW**

```
Page Loads
    ↓
index.html loads detection-engine.js
    ↓
detection-engine.js runs:
    - Gets all HTML elements (video, canvas, buttons)
    - Shows homepage screen
    - Initializes audio system
    ↓
User sees homepage with toggle & start button
```

### **2. USER INTERACTION FLOW**

```
Homepage
    ↓
User toggles "Enable Phone Detection" ON
    ↓
"Start Monitoring" button becomes enabled
    ↓
User clicks "Start Monitoring"
    ↓
Permission Screen appears
    ↓
User clicks "Grant Camera Access"
    ↓
Browser requests camera permission
    ↓
If granted:
    - Video stream starts
    - Detection screen appears
    - startSystem() is called
```

### **3. SYSTEM STARTUP (startSystem function)**

```
startSystem() called
    ↓
Step 1: Load ONNX.js library
    - Creates <script> tag
    - Loads from CDN
    - Waits for window.ort to be available
    ↓
Step 2: Initialize ONNX Runtime
    - Sets WASM to single thread
    - Disables SIMD
    ↓
Step 3: Load YOLOv5 Model
    - Reads yolov5n.onnx file
    - Creates InferenceSession
    - Model ready for predictions
    ↓
Step 4: Wait for video stream
    - Checks every 100ms if video is ready
    - When ready, starts detection loop
```

### **4. DETECTION LOOP (predictWebcam function)**

This runs every 2 seconds:

```
predictWebcam() called
    ↓
Check 1: Is system running? → No? Exit
Check 2: Is video ready? → No? Retry in 500ms
Check 3: Has 2 seconds passed? → No? Retry in 500ms
Check 4: Is already processing? → Yes? Retry in 500ms
    ↓
All checks pass → Start processing
    ↓
Yield to browser (setTimeout 200ms)
    ↓
PREPROCESSING:
    - Create temporary canvas
    - Draw video frame to canvas (resize to 640x640)
    - Get pixel data
    - Convert RGB to BGR
    - Normalize values (0-255 → 0-1)
    - Create Float32Array tensor
    ↓
Yield to browser (10ms)
    ↓
CREATE TENSOR:
    - Shape: [1, 3, 640, 640]
    - 1 = batch size
    - 3 = RGB channels
    - 640x640 = image size
    ↓
Yield to browser (10ms)
    ↓
RUN AI MODEL:
    - model.run(feeds)
    - This is the BLOCKING operation
    - Takes 100-500ms
    - Returns detection results
    ↓
POSTPROCESSING:
    - Parse output array
    - Extract bounding boxes
    - Filter by confidence (15% threshold)
    - Filter by class (only class 67 = cell phone)
    - Apply Non-Maximum Suppression (remove duplicates)
    - Scale coordinates to video size
    ↓
Yield to browser (10ms)
    ↓
DISPLAY RESULTS:
    - Clear canvas
    - Draw red boxes around phones
    - Update status panel
    - Play sound if phone found
    - Show notification if tab inactive
    ↓
Schedule next detection (1000ms delay)
```

### **5. DATA FLOW**

```
Video Stream (Webcam)
    ↓
<video> element displays live feed
    ↓
predictWebcam() captures frame
    ↓
preprocess() converts to tensor
    ↓
YOLOv5 Model processes tensor
    ↓
Model outputs: [1, 25200, 85]
    - 25200 = possible detections
    - 85 = [x, y, w, h, conf, 80 class scores]
    ↓
postprocess() filters results
    ↓
Only phones (class 67) with >15% confidence
    ↓
displayDetections() draws boxes
    ↓
<canvas> overlay shows boxes
    ↓
updateStatus() updates UI
```

---

## 🔗 **How Components Connect**

### **HTML → JavaScript**

```javascript
// HTML defines elements:
<button id="start-btn-alt">Start Monitoring</button>
<video id="webcam"></video>
<canvas id="canvas-overlay"></canvas>

// JavaScript gets references:
const startBtnAlt = document.getElementById('start-btn-alt');
const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas-overlay');
```

### **Event Listeners Chain**

```
Toggle Change Event
    ↓
syncToggles() updates both toggles
    ↓
Enables/disables start buttons
    ↓
Start Button Click
    ↓
showScreen('permission')
    ↓
Permission Button Click
    ↓
getUserMedia() requests camera
    ↓
startSystem() loads AI model
    ↓
predictWebcam() starts detection loop
```

### **State Management**

```javascript
// Global state variables:
let model = null;              // AI model instance
let isRunning = false;         // Detection active?
let isProcessing = false;      // Currently processing?
let lastPredictionTime = 0;    // Rate limiting
let soundEnabled = true;       // Sound alerts on/off
```

---

## 🎨 **UI Screen System**

### **Three Screens (only one visible at a time)**

1. **Homepage** (`#homepage`)
   - Marketing content
   - Toggle & start button
   - Features section

2. **Permission Screen** (`#permission-screen`)
   - Camera permission request
   - Privacy notice

3. **Detection Screen** (`#detection-screen`)
   - Live video feed
   - Canvas overlay (boxes)
   - Status panel
   - Stop button

**Switching screens:**
```javascript
showScreen('homepage')    // Shows homepage, hides others
showScreen('permission')  // Shows permission, hides others
showScreen('detection')   // Shows detection, hides others
```

---

## 🤖 **AI Detection Process**

### **YOLOv5 Model**

- **Input**: 640x640 RGB image (normalized 0-1)
- **Output**: Array of detections
- **Format**: [batch, num_boxes, 85]
  - Each box has: x, y, width, height, confidence, 80 class scores
- **Class 67** = "cell phone" in COCO dataset

### **Detection Pipeline**

```
Raw Video Frame
    ↓
Resize to 640x640
    ↓
RGB → BGR conversion
    ↓
Normalize (0-1)
    ↓
Tensor [1, 3, 640, 640]
    ↓
YOLOv5 Inference
    ↓
Raw Output [1, 25200, 85]
    ↓
Filter by confidence (>10%)
    ↓
Find best class for each box
    ↓
Filter by class (only 67)
    ↓
Filter by final score (>15%)
    ↓
Non-Maximum Suppression
    ↓
Scale to video dimensions
    ↓
Draw boxes on canvas
```

---

## 🔊 **Sound & Notification System**

### **Sound Alert**

```javascript
playAlertSound()
    ↓
Checks: soundEnabled? audioContext exists?
    ↓
Checks cooldown (2 seconds)
    ↓
Creates Web Audio oscillator
    ↓
Plays 800Hz beep for 0.5 seconds
    ↓
Works even when tab is inactive!
```

### **Browser Notification**

```javascript
updateStatus() detects phone
    ↓
Checks: tab hidden? permission granted?
    ↓
Creates browser notification
    ↓
Shows even when tab is inactive
```

---

## ⚡ **Performance Optimizations**

### **Non-Blocking Design**

1. **Rate Limiting**: Only runs every 2 seconds
2. **Yielding**: Uses setTimeout between steps
3. **requestIdleCallback**: Runs when browser is idle
4. **Single Thread**: WASM uses 1 thread to prevent blocking
5. **Overlap Prevention**: `isProcessing` flag prevents concurrent runs

### **Why This Matters**

- Without these: Page freezes, can't click anything
- With these: Page stays responsive, smooth experience

---

## 🔄 **Complete User Journey**

```
1. User opens website
   → Homepage loads
   → JavaScript initializes

2. User reads content
   → Sees hero, about, features

3. User enables toggle
   → Toggle syncs
   → Start button enables

4. User clicks "Start Monitoring"
   → Permission screen shows
   → User grants camera access

5. System starts
   → Loads ONNX.js (from CDN)
   → Loads YOLOv5 model (from file)
   → Starts video stream

6. Detection begins
   → Every 2 seconds:
     - Captures frame
     - Preprocesses
     - Runs AI
     - Postprocesses
     - Displays results

7. Phone detected
   → Red box drawn
   → Sound plays
   → Status updates
   → Notification shows (if tab inactive)

8. User clicks "Stop"
   → Stops video stream
   → Returns to homepage
   → Resets toggles
```

---

## 🛠️ **Key Technologies**

- **ONNX.js**: Runs AI models in browser
- **YOLOv5 Nano**: Lightweight object detection model
- **WebRTC**: Camera access (getUserMedia)
- **Canvas API**: Drawing detection boxes
- **Web Audio API**: Sound alerts
- **Notifications API**: Browser notifications

---

## 📊 **Data Structures**

### **Detection Object**
```javascript
{
    x: 100,           // Top-left X coordinate
    y: 200,           // Top-left Y coordinate
    width: 150,       // Box width
    height: 200,      // Box height
    class: 67,        // COCO class ID
    className: "cell phone",
    score: 0.85       // Confidence (0-1)
}
```

### **Model Input**
```javascript
Tensor {
    type: 'float32',
    data: Float32Array[1228800],  // 3 * 640 * 640
    dims: [1, 3, 640, 640]
}
```

---

## 🎯 **Summary**

**Everything connects through:**
1. **HTML** provides structure and elements
2. **JavaScript** gets elements and adds event listeners
3. **Events** trigger functions (clicks, toggles)
4. **Functions** call other functions in sequence
5. **AI Model** processes video frames
6. **Results** update UI and trigger alerts
7. **Loop** continues every 2 seconds

**The magic happens in `predictWebcam()` - it's the heart of the system!**

