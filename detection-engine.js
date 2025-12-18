// YOLOv8 Nano Detection Engine - Non-blocking version
// Supports both YOLOv5 and YOLOv8 models (auto-detected)
// Load ONNX.js from CDN
let ort = null;

// Load ONNX.js library
async function loadONNX() {
    if (typeof window !== 'undefined' && window.ort) {
        return window.ort;
    }
    
    // Try loading from CDN - use version 1.18 for better compatibility
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js';
    script.async = true;
    
    return new Promise((resolve, reject) => {
        script.onload = () => {
            if (window.ort) {
                // Configure WASM with more memory
                if (window.ort.env && window.ort.env.wasm) {
                    window.ort.env.wasm.numThreads = 1;
                    window.ort.env.wasm.simd = true;
                }
                resolve(window.ort);
            } else {
                reject(new Error('ONNX.js failed to load'));
            }
        };
        script.onerror = () => reject(new Error('Failed to load ONNX.js script'));
        document.head.appendChild(script);
    });
}

// UI Elements
const homepage = document.getElementById('homepage');
const permissionScreen = document.getElementById('permission-screen');
const detectionScreen = document.getElementById('detection-screen');
const detectionToggle = document.getElementById('detection-toggle');
const detectionToggleAlt = document.getElementById('detection-toggle-alt');
const startBtn = document.getElementById('start-btn');
const startBtnAlt = document.getElementById('start-btn-alt');
const requestPermissionBtn = document.getElementById('request-permission-btn');
const stopBtn = document.getElementById('stop-btn');
const soundToggle = document.getElementById('sound-toggle');

const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas-overlay');
const ctx = canvas.getContext('2d');
const statusPanel = document.getElementById('status-panel');

let model = null;
let isRunning = false; // Start as false, only run when user enables
let isProcessing = false;
let lastPredictionTime = 0;
let lastPhoneDetectionTime = 0;
const DETECTION_INTERVAL = 2000; // 2 seconds - more frequent detection
const SOUND_COOLDOWN = 2000; // 2 seconds between sound alerts

// Sound notification system (works even when tab is inactive)
let audioContext = null;
let soundEnabled = true;

function initAudio() {
    try {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    } catch (e) {
        console.warn('Audio context not supported:', e);
    }
}

function playAlertSound() {
    if (!soundEnabled || !audioContext) return;
    
    // Check cooldown to prevent spam
    const now = Date.now();
    if (now - lastPhoneDetectionTime < SOUND_COOLDOWN) return;
    lastPhoneDetectionTime = now;
    
    try {
        // Create a beep sound using Web Audio API
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        oscillator.frequency.value = 800; // Higher pitch
        oscillator.type = 'sine';
        
        gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
        
        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + 0.5);
        
        // Resume audio context if suspended (required by some browsers)
        if (audioContext.state === 'suspended') {
            audioContext.resume();
        }
    } catch (e) {
        console.warn('Failed to play sound:', e);
    }
}

// COCO class names (YOLOv5 uses COCO dataset)
const CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
];

// Phone class index in COCO (cell phone = 67)
const PHONE_CLASS_INDEX = 67;

async function startSystem() {
    statusPanel.innerText = "Loading ONNX Runtime...";

    try {
        // Load ONNX.js library first
        ort = await loadONNX();
        console.log('ONNX.js loaded successfully');
        
        // Initialize ONNX Runtime - check if env exists
        if (ort && ort.env && ort.env.wasm) {
            ort.env.wasm.numThreads = 1; // Single thread to prevent blocking
            ort.env.wasm.simd = false; // Disable SIMD for compatibility
        } else {
            console.warn('ONNX.js env.wasm not available, using defaults');
        }
        
        statusPanel.innerText = "Loading YOLOv8 Model...";
        
        // Load YOLO ONNX model from local file
        // Supports both YOLOv5 and YOLOv8 (auto-detected by output format)
        const modelUrl = './yolov8n.onnx'; // YOLOv8 Nano - 33% more accurate than v5!
        
        if (!ort || !ort.InferenceSession) {
            throw new Error('ONNX.js library not loaded properly');
        }
        
        try {
            console.log('Loading YOLO model from:', modelUrl);
            
            // Fetch the model as ArrayBuffer (more reliable for large models)
            statusPanel.innerText = "Downloading model...";
            const response = await fetch(modelUrl);
            if (!response.ok) {
                throw new Error(`Model file not found (HTTP ${response.status})`);
            }
            
            const contentLength = response.headers.get('content-length');
            console.log('Model file found, size:', contentLength, 'bytes');
            
            const modelBuffer = await response.arrayBuffer();
            console.log('Model downloaded, creating session...');
            statusPanel.innerText = "Initializing AI...";
            
            // Load from ArrayBuffer
            model = await ort.InferenceSession.create(modelBuffer, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'basic',
            });
            statusPanel.innerText = "Model Loaded Successfully!";
            console.log('Model loaded! Input names:', model.inputNames, 'Output names:', model.outputNames);
        } catch (modelError) {
            console.error('Model load error:', modelError);
            const errorMsg = modelError.message || modelError.toString() || 'Unknown error';
            throw new Error(`Failed to load model: ${errorMsg}. Make sure yolov8n.onnx is in the project folder.`);
        }

        statusPanel.innerText = "Model Ready - Waiting for camera...";
        
        // Wait for video stream to be set (will be set by permission handler)
        const checkVideo = setInterval(() => {
            if (video.srcObject && video.readyState >= 2) {
                clearInterval(checkVideo);
                isRunning = true;
                statusPanel.innerText = "Active: YOLOv8 Nano";
                statusPanel.classList.add('status-safe');
                // Start detection with delay
                setTimeout(() => {
                    if (window.requestIdleCallback) {
                        requestIdleCallback(predictWebcam, { timeout: 1000 });
                    } else {
                        setTimeout(predictWebcam, 1000);
                    }
                }, 500);
            }
        }, 100);
        
        // Set timeout to stop checking after 30 seconds
        setTimeout(() => clearInterval(checkVideo), 30000);

    } catch (err) {
        statusPanel.innerText = "Error: " + err.message;
        console.error("Startup error:", err);
    }
}

// Preprocess image for YOLO models
function preprocess(imageData, modelWidth, modelHeight) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = modelWidth;
    canvas.height = modelHeight;
    
    ctx.drawImage(imageData, 0, 0, modelWidth, modelHeight);
    const imageDataProcessed = ctx.getImageData(0, 0, modelWidth, modelHeight);
    
    const input = new Float32Array(3 * modelWidth * modelHeight);
    const data = imageDataProcessed.data;
    
    // Normalize to [0, 1] - YOLO expects RGB (not BGR!)
    for (let i = 0; i < data.length; i += 4) {
        const r = data[i] / 255.0;
        const g = data[i + 1] / 255.0;
        const b = data[i + 2] / 255.0;
        
        const index = Math.floor(i / 4);
        input[index] = r; // R (was incorrectly B)
        input[index + modelWidth * modelHeight] = g; // G
        input[index + 2 * modelWidth * modelHeight] = b; // B (was incorrectly R)
    }
    
    return input;
}

// Post-process YOLO output (supports both YOLOv5 and YOLOv8)
function postprocess(output, imgWidth, imgHeight, modelWidth, modelHeight) {
    const detections = [];
    const outputDims = output.dims;
    const outputData = Array.from(output.data);
    
    const scaleX = imgWidth / modelWidth;
    const scaleY = imgHeight / modelHeight;
    
    console.log('Output dims:', outputDims);
    
    // YOLOv8 format: [1, 84, 8400] - transposed!
    // 84 = 4 (x,y,w,h) + 80 (class scores)
    // 8400 = number of detection candidates
    if (outputDims.length === 3 && outputDims[1] === 84) {
        console.log('Detected YOLOv8 output format');
        const numClasses = 80;
        const numDetections = outputDims[2]; // 8400
        
        for (let i = 0; i < numDetections; i++) {
            // YOLOv8 is transposed: data is stored as [feature][detection]
            const x_center = outputData[0 * numDetections + i] * scaleX;
            const y_center = outputData[1 * numDetections + i] * scaleY;
            const w = outputData[2 * numDetections + i] * scaleX;
            const h = outputData[3 * numDetections + i] * scaleY;
            
            // Find class with highest score (no separate objectness in YOLOv8)
            let maxClass = 0;
            let maxScore = outputData[4 * numDetections + i]; // First class score
            
            for (let j = 1; j < numClasses; j++) {
                const score = outputData[(4 + j) * numDetections + i];
                if (score > maxScore) {
                    maxScore = score;
                    maxClass = j;
                }
            }
            
            // Skip low confidence detections
            if (maxScore < 0.4) continue;
            
            // ONLY DETECT PHONES (class 67 = cell phone in COCO)
            if (maxClass !== 67) continue;
            
            console.log(`YOLOv8 Phone detected! Confidence: ${(maxScore * 100).toFixed(1)}%`);
            
            detections.push({
                x: x_center - w / 2,
                y: y_center - h / 2,
                width: w,
                height: h,
                class: maxClass,
                className: CLASS_NAMES[maxClass] || `class_${maxClass}`,
                score: maxScore
            });
        }
        
        return nms(detections, 0.45);
    }
    
    // YOLOv5 format: [1, 25200, 85]
    // 85 = 4 (x,y,w,h) + 1 (objectness) + 80 (class scores)
    if (outputDims.length === 3 && outputDims[2] === 85) {
        console.log('Detected YOLOv5 output format');
        const numBoxes = outputDims[1];
        
        for (let i = 0; i < numBoxes; i++) {
            const offset = i * 85;
            const x_center = outputData[offset] * scaleX;
            const y_center = outputData[offset + 1] * scaleY;
            const w = outputData[offset + 2] * scaleX;
            const h = outputData[offset + 3] * scaleY;
            const conf = outputData[offset + 4]; // Objectness score
            
            if (conf < 0.4) continue;
            
            // Find class with highest score
            let maxClass = 0;
            let maxScore = outputData[offset + 5];
            for (let j = 1; j < 80; j++) {
                if (outputData[offset + 5 + j] > maxScore) {
                    maxScore = outputData[offset + 5 + j];
                    maxClass = j;
                }
            }
            
            const finalScore = conf * maxScore;
            if (finalScore < 0.4) continue;
            
            // ONLY DETECT PHONES (class 67 = cell phone)
            if (maxClass !== 67) continue;
            
            console.log(`YOLOv5 Phone detected! Confidence: ${(finalScore * 100).toFixed(1)}%`);
            
            detections.push({
                x: x_center - w / 2,
                y: y_center - h / 2,
                width: w,
                height: h,
                class: maxClass,
                className: CLASS_NAMES[maxClass] || `class_${maxClass}`,
                score: finalScore
            });
        }
        
        return nms(detections, 0.45);
    }
    
    // Fallback for other formats
    console.log('Unknown output format:', outputDims);
    return detections;
}

// Simple NMS implementation
function nms(detections, iouThreshold) {
    detections.sort((a, b) => b.score - a.score);
    const filtered = [];
    
    while (detections.length > 0) {
        const best = detections.shift();
        filtered.push(best);
        detections = detections.filter(det => {
            const iou = calculateIOU(best, det);
            return iou < iouThreshold;
        });
    }
    
    return filtered;
}

function calculateIOU(box1, box2) {
    const x1 = Math.max(box1.x, box2.x);
    const y1 = Math.max(box1.y, box2.y);
    const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
    const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);
    
    if (x2 < x1 || y2 < y1) return 0;
    
    const intersection = (x2 - x1) * (y2 - y1);
    const area1 = box1.width * box1.height;
    const area2 = box2.width * box2.height;
    const union = area1 + area2 - intersection;
    
    return intersection / union;
}

async function predictWebcam() {
    if (!isRunning || !model || !ort) return;
    
    // Check if video is ready
    if (!video || video.videoWidth === 0 || video.readyState < 2) {
        setTimeout(predictWebcam, 500);
        return;
    }

    // Rate limiting - only run every 2 seconds
    const now = Date.now();
    if (now - lastPredictionTime < DETECTION_INTERVAL) {
        setTimeout(predictWebcam, 500);
        return;
    }

    // Prevent overlapping
    if (isProcessing) {
        setTimeout(predictWebcam, 500);
        return;
    }

    lastPredictionTime = now;
    isProcessing = true;

    // Resize canvas to match video exactly
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    }

    // Use setTimeout to yield to browser BEFORE processing
    // This ensures the page can handle clicks/interactions
    setTimeout(async () => {
        try {
            const modelWidth = 640;
            const modelHeight = 640;
            
            // Yield to browser multiple times during processing
            // Preprocess
            await new Promise(resolve => setTimeout(resolve, 10));
            const input = preprocess(video, modelWidth, modelHeight);
            
            // Create tensor
            await new Promise(resolve => setTimeout(resolve, 10));
            const tensor = new ort.Tensor('float32', input, [1, 3, modelHeight, modelWidth]);
            
            // Run inference - this is the blocking part
            // But we've yielded multiple times already, so browser can handle events
            const feeds = { [model.inputNames[0]]: tensor };
            const results = await model.run(feeds);
            
            // Get output
            const output = results[model.outputNames[0]];
            
            // Post-process
            await new Promise(resolve => setTimeout(resolve, 10));
            const detections = postprocess(output, video.videoWidth, video.videoHeight, modelWidth, modelHeight);
            
            // Debug: Log all detections
            if (detections.length > 0) {
                console.log(`Found ${detections.length} phone detection(s):`, detections);
            }
            
            // Render - use requestAnimationFrame to ensure smooth updates
            requestAnimationFrame(() => {
                displayDetections(detections);
                isProcessing = false;
            });
            
        } catch (err) {
            console.warn("Detection error:", err);
            isProcessing = false;
        }
        
        // Schedule next detection with longer delay
        setTimeout(predictWebcam, 1000);
    }, 200); // Initial delay to yield to browser
}

function displayDetections(detections) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    let phoneFound = false;

    // Filter to only phones (should already be filtered, but double-check)
    const phoneDetections = detections.filter(det => det.class === 67 || det.className === 'cell phone');

    phoneDetections.forEach((det) => {
        const { x, y, width, height, className, score } = det;
        const scorePercent = Math.round(score * 100);

        phoneFound = true;

        // Canvas is mirrored with CSS transform: scaleX(-1) to match video
        // So we draw at the original coordinates (no mirroring needed in code)
        // Ensure coordinates are within canvas bounds
        const boxX = Math.max(0, Math.min(x, canvas.width - width));
        const boxY = Math.max(0, Math.min(y, canvas.height - height));
        const boxW = Math.min(width, canvas.width - boxX);
        const boxH = Math.min(height, canvas.height - boxY);
        
        drawBox(boxX, boxY, boxW, boxH, className, scorePercent);
    });

    updateStatus(phoneFound);
}

function drawBox(x, y, w, h, label, score) {
    ctx.strokeStyle = '#00FF00';
    if (label === 'cell phone') {
        ctx.strokeStyle = '#FF0000';
    }

    ctx.lineWidth = 4;
    ctx.strokeRect(x, y, w, h);
    
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fillRect(x, y, w, 25);
    
    ctx.fillStyle = '#FFFFFF';
    ctx.font = "20px Arial";
    ctx.fillText(`${label} ${score}%`, x + 5, y + 20);
}

function updateStatus(found) {
    if (found) {
        statusPanel.innerText = "⚠️ PHONE DETECTED ⚠️";
        statusPanel.classList.remove('status-safe');
        statusPanel.classList.add('status-danger');
        
        // Play sound alert
        playAlertSound();
        
        // Show browser notification if tab is not active
        if (document.hidden && Notification.permission === 'granted') {
            new Notification('Phone Detected!', {
                body: 'A phone has been detected. Stay focused!',
                icon: '📱',
                tag: 'phone-detection'
            });
        }
    } else {
        statusPanel.innerText = "Scanning...";
        statusPanel.classList.remove('status-danger');
        statusPanel.classList.add('status-safe');
    }
}

// UI Flow Management
function showScreen(screenName) {
    homepage.classList.remove('active');
    permissionScreen.classList.remove('active');
    detectionScreen.classList.remove('active');
    
    if (screenName === 'homepage') {
        homepage.classList.add('active');
    } else if (screenName === 'permission') {
        permissionScreen.classList.add('active');
    } else if (screenName === 'detection') {
        detectionScreen.classList.add('active');
    }
}

// Event Listeners - Sync both toggle buttons
function syncToggles(isChecked) {
    if (detectionToggle) detectionToggle.checked = isChecked;
    if (detectionToggleAlt) detectionToggleAlt.checked = isChecked;
    if (startBtn) startBtn.disabled = !isChecked;
    if (startBtnAlt) startBtnAlt.disabled = !isChecked;
}

if (detectionToggle) {
    detectionToggle.addEventListener('change', (e) => {
        syncToggles(e.target.checked);
    });
}

if (detectionToggleAlt) {
    detectionToggleAlt.addEventListener('change', (e) => {
        syncToggles(e.target.checked);
    });
}

if (startBtn) {
    startBtn.addEventListener('click', () => {
        showScreen('permission');
    });
}

if (startBtnAlt) {
    startBtnAlt.addEventListener('click', () => {
        showScreen('permission');
    });
}

requestPermissionBtn.addEventListener('click', async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: 10 }
            } 
        });
        video.srcObject = stream;
        showScreen('detection');
        initAudio();
        
        // Request notification permission
        if ('Notification' in window && Notification.permission === 'default') {
            Notification.requestPermission();
        }
        
        // Start the detection system (model loading)
        await startSystem();
        
        // Wait for video to load
        video.addEventListener("loadeddata", () => {
            isRunning = true;
            statusPanel.innerText = "Active: YOLOv8 Nano";
            statusPanel.classList.add('status-safe');
            setTimeout(() => {
                if (window.requestIdleCallback) {
                    requestIdleCallback(predictWebcam, { timeout: 1000 });
                } else {
                    setTimeout(predictWebcam, 1000);
                }
            }, 500);
        }, { once: true });
    } catch (err) {
        alert('Camera access denied. Please allow camera access to use Focus Guard.');
        showScreen('homepage');
    }
});

stopBtn.addEventListener('click', () => {
    isRunning = false;
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
        video.srcObject = null;
    }
    showScreen('homepage');
    syncToggles(false);
});

soundToggle.addEventListener('change', (e) => {
    soundEnabled = e.target.checked;
});

// Initialize on page load
showScreen('homepage');
initAudio();
