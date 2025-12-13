// 1. IMPORT LIBRARY DIRECTLY (Fixes "cocoSsd is not defined")
import { 
    ObjectDetector, 
    FilesetResolver 
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas-overlay');
const ctx = canvas.getContext('2d');
const statusPanel = document.getElementById('status-panel');

let objectDetector = undefined;
let runningMode = "VIDEO";
let lastVideoTime = -1;

async function startSystem() {
    statusPanel.innerText = "Loading AI...";

    try {
        // 2. LOAD ENGINE
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
        );

        // 3. LOAD MODEL (EfficientDet-Lite0 - Recommended)
        objectDetector = await ObjectDetector.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: "https://storage.googleapis.com/mediapipe-tasks/object_detector/efficientdet_lite0_uint8.tflite",
                delegate: "GPU"
            },
            scoreThreshold: 0.3, // 30% Sensitivity
            runningMode: runningMode,
            categoryAllowlist: ["cell phone", "mobile phone"] // ONLY look for phones
        });

        statusPanel.innerText = "Accessing Camera...";
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
        
        statusPanel.innerText = "System Active";
        statusPanel.classList.add('status-safe');

    } catch (err) {
        console.error("Startup Error:", err);
        statusPanel.innerText = "Error: See Console";
    }
}

async function predictWebcam() {
    
    if (!video || video.videoWidth === 0 || video.videoHeight === 0) {
        window.requestAnimationFrame(predictWebcam);
        return;
    }

   
    if (canvas.width !== video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    }

    
    if (video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        
        if (objectDetector) {
            const startTimeMs = performance.now();
            const detections = objectDetector.detectForVideo(video, startTimeMs).detections;
            displayVideoDetections(detections);
        }
    }
    
  
    window.requestAnimationFrame(predictWebcam);
}

function displayVideoDetections(detections) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    let phoneFound = false;

    detections.forEach((result) => {
        phoneFound = true; // Anything in this list IS a phone
        
        const { originX, originY, width, height } = result.boundingBox;
        const category = result.categories[0];
        const score = Math.round(category.score * 100);

        const mirroredX = canvas.width - originX - width;

        drawBox(mirroredX, originY, width, height, "PHONE", score);
    });

    updateStatus(phoneFound);
}

function drawBox(x, y, w, h, label, score) {
    ctx.strokeStyle = '#FF0000';
    ctx.lineWidth = 4;
    ctx.strokeRect(x, y, w, h);
    
    ctx.fillStyle = '#FF0000';
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
    } else {
        statusPanel.innerText = "Status: Clear";
        statusPanel.classList.remove('status-danger');
        statusPanel.classList.add('status-safe');
    }
}

startSystem();