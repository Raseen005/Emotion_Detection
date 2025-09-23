// Function to get emotion colors from the CSS
function getEmotionColors() {
    return {
        'Anger': '#e74c3c',
        'Contempt': '#c0392b',
        'Disgust': '#27ae60',
        'Fear': '#8e44ad',
        'Happy': '#f39c12',
        'Neutral': '#95a5a6',
        'Sad': '#3498db',
        'Surprise': '#e67e22',
    };
}

// Map emotions to emojis
const EMOJI_MAP = {
    'Anger': '😠',
    'Contempt': '😒',
    'Disgust': '🤢',
    'Fear': '😨',
    'Happy': '😊',
    'Neutral': '😐',
    'Sad': '😢',
    'Surprise': '😮',
};

class EmotionDetectionApp {
    constructor() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
        this.stream = null;
        this.autoDetectInterval = null;
        this.isDetecting = false;
        this.emotionColors = getEmotionColors();

        this.initializeEventListeners();
    }

    initializeEventListeners() {
        document.getElementById('startBtn').addEventListener('click', () => this.startCamera());
        document.getElementById('stopBtn').addEventListener('click', () => this.stopCamera());
        document.getElementById('autoDetect').addEventListener('change', (e) => this.toggleAutoDetect(e.target.checked));
    }

    startCamera() {
        if (this.stream) return;

        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                this.stream = stream;
                this.video.srcObject = stream;
                this.video.onloadedmetadata = () => {
                    this.video.play();
                    this.canvas.width = this.video.videoWidth;
                    this.canvas.height = this.video.videoHeight;
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    document.getElementById('autoDetect').checked = true;
                    this.toggleAutoDetect(true);
                };
            })
            .catch(error => {
                console.error("Error accessing the camera:", error);
                alert("Failed to access camera. Please check permissions.");
            });
    }

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        this.video.srcObject = null;
        this.toggleAutoDetect(false);
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        document.getElementById('startBtn').disabled = false;
        document.getElementById('stopBtn').disabled = true;
    }

    toggleAutoDetect(checked) {
        if (checked) {
            if (!this.autoDetectInterval) {
                this.isDetecting = true;
                this.autoDetectInterval = setInterval(() => this.captureAndAnalyze(), 100);
            }
        } else {
            if (this.autoDetectInterval) {
                clearInterval(this.autoDetectInterval);
                this.autoDetectInterval = null;
                this.isDetecting = false;
            }
        }
    }

    captureAndAnalyze() {
        if (!this.video.paused && !this.video.ended && this.isDetecting) {
            this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            this.canvas.toBlob(blob => {
                const reader = new FileReader();
                reader.onloadend = () => {
                    this.sendFrameToServer(reader.result);
                };
                reader.readAsDataURL(blob);
            }, 'image/jpeg');
        }
    }

    async sendFrameToServer(imageDataUrl) {
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image: imageDataUrl })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.updateUI(data.faces);

        } catch (error) {
            console.error("Error communicating with the backend:", error);
            this.updateUI([]);
        }
    }

    updateUI(faces) {
        const emotionText = document.getElementById('emotionText');
        const emotionEmoji = document.getElementById('emotionEmoji');
        
        if (faces.length > 0) {
            const firstFace = faces[0];
            const emotion = firstFace.emotion;
            emotionText.textContent = emotion;
            emotionEmoji.textContent = EMOJI_MAP[emotion] || '';
        } else {
            emotionText.textContent = 'Detecting...';
            emotionEmoji.textContent = '';
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new EmotionDetectionApp();
});