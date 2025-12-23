"""
THUMBS_UP = Select highlighted card
SWIPE = Navigate between cards
FIST = Go back / Close

"""

import asyncio
import json
import urllib.request
from pathlib import Path
import webbrowser

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

try:
    import websockets
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "websockets"])
    import websockets

from gesture_classifier import Gesture, GestureClassifier


def landmarks_to_array(hand_landmarks) -> np.ndarray:
    arr = np.zeros((21, 3))
    for i, lm in enumerate(hand_landmarks):
        arr[i] = [lm.x, lm.y, lm.z]
    return arr


class GestureServer:
    def __init__(self):
        self.classifier = GestureClassifier()
        self.cap = None
        self.detector = None
        self.latest_result = None
        self.running = False
        self.clients = set()
        self.cooldown = 0

    def _callback(self, result, output_image, timestamp_ms):
        self.latest_result = result

    def start(self):
        model_path = Path("hand_landmarker.task")
        if not model_path.exists():
            print("Downloading model...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            result_callback=self._callback
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")
        self.running = True

    def process_frame(self, ts):
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.detector.detect_async(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        result = self.latest_result
        gesture = Gesture.NONE
        bends = {}
        landmarks_2d = None

        if result and result.hand_landmarks:
            lm = result.hand_landmarks[0]
            arr = landmarks_to_array(lm)
            gesture, bends = self.classifier.classify(arr)
            landmarks_2d = [[p.x, p.y] for p in lm]

            h, w = frame.shape[:2]
            color = (0, 255, 255) if gesture != Gesture.NONE else (0, 255, 0)
            for p in lm:
                cv2.circle(frame, (int(p.x * w), int(p.y * h)), 5, color, -1)

            conns = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
                     (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
            for s, e in conns:
                cv2.line(frame, (int(lm[s].x*w), int(lm[s].y*h)), (int(lm[e].x*w), int(lm[e].y*h)), (0,200,0), 2)

        # Debug display
        cv2.putText(frame, f"GESTURE: {gesture.value}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                   (0,255,255) if gesture != Gesture.NONE else (200,200,200), 2)

        y = 60
        for f in ["index", "middle", "ring", "pinky"]:
            if f in bends:
                b = bends[f]
                cv2.rectangle(frame, (10, y), (110, y+12), (50,50,50), -1)
                cv2.rectangle(frame, (10, y), (10+int(b*100), y+12), (0,0,255) if b > 0.45 else (0,255,0), -1)
                cv2.putText(frame, f"{f}: {'BENT' if b>0.45 else 'OPEN'}", (120, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
                y += 18

        cv2.putText(frame, "SWIPE L/R = Move | THUMBS UP = Select | FIST = Back", (10, frame.shape[0]-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

        cv2.imshow("Kiosk Debug", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.running = False

        return {
            "gesture": gesture.value, 
            "hand_detected": landmarks_2d is not None, 
            "landmarks": landmarks_2d
        }

    async def broadcast(self, msg):
        if self.clients:
            await asyncio.gather(*[c.send(json.dumps(msg)) for c in self.clients])

    async def capture_loop(self):
        ts = 0
        while self.running:
            ts += 33
            data = self.process_frame(ts)
            if data:
                if data["gesture"] != "NONE" and self.cooldown <= 0:
                    await self.broadcast(data)
                    print(f">>> {data['gesture']}")
                    self.cooldown = 15
                elif data["hand_detected"]:
                    await self.broadcast({"gesture": "NONE", "hand_detected": True, "landmarks": data["landmarks"]})
                    
                if self.cooldown > 0:
                    self.cooldown -= 1
            await asyncio.sleep(0.016)

    async def handle_client(self, ws):
        self.clients.add(ws)
        print(f"Client connected ({len(self.clients)})")
        try:
            await ws.wait_closed()
        finally:
            self.clients.remove(ws)

    def stop(self):
        self.running = False
        if self.detector:
            self.detector.close()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


KIOSK_HTML = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Touchless Hospital Kiosk</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Segoe UI', Arial; }
        body { background: linear-gradient(135deg, #1a1a2e, #16213e); min-height: 100vh; color: white; overflow: hidden; }
        
        .header { background: rgba(255,255,255,0.1); padding: 12px 25px; display: flex; justify-content: space-between; align-items: center; }
        .logo { font-size: 20px; font-weight: bold; }
        .logo span { color: #4ecca3; }
        .status { display: flex; align-items: center; gap: 8px; font-size: 14px; }
        .status-dot { width: 10px; height: 10px; border-radius: 50%; background: #ff6b6b; }
        .status-dot.ok { background: #4ecca3; }
        
        .main { padding: 20px 30px; }
        
        .gesture-bar { 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            gap: 15px; 
            margin-bottom: 20px;
            background: rgba(0,0,0,0.3);
            padding: 10px 20px;
            border-radius: 30px;
            width: fit-content;
            margin-left: auto;
            margin-right: auto;
        }
        .gesture-icon { font-size: 30px; }
        .gesture-name { font-size: 16px; color: #4ecca3; }
        
        h2 { text-align: center; margin-bottom: 20px; font-size: 22px; }
        
        .cards { 
            display: grid; 
            grid-template-columns: repeat(3, 1fr); 
            gap: 15px; 
            max-width: 900px;
            margin: 0 auto;
        }
        
        .card { 
            background: rgba(255,255,255,0.08); 
            border-radius: 15px; 
            padding: 25px 15px; 
            text-align: center; 
            border: 3px solid transparent; 
            transition: all 0.2s;
            cursor: pointer;
        }
        .card.selected { 
            background: rgba(78,204,163,0.3); 
            border-color: #4ecca3; 
            transform: scale(1.05);
            box-shadow: 0 0 30px rgba(78,204,163,0.4);
        }
        .card-icon { font-size: 40px; margin-bottom: 10px; }
        .card-title { font-size: 16px; font-weight: bold; margin-bottom: 5px; }
        .card-desc { font-size: 12px; color: rgba(255,255,255,0.6); }
        
        .card-nav { 
            display: flex; 
            justify-content: center; 
            align-items: center;
            gap: 20px;
            margin-top: 20px;
        }
        .card-counter {
            font-size: 14px;
            color: rgba(255,255,255,0.6);
        }
        
        .instructions { 
            position: fixed; 
            bottom: 15px; 
            left: 50%; 
            transform: translateX(-50%); 
            background: rgba(0,0,0,0.7); 
            padding: 12px 25px; 
            border-radius: 25px; 
            display: flex; 
            gap: 25px; 
            font-size: 13px; 
        }
        .instruction { display: flex; align-items: center; gap: 6px; }
        .instruction span { font-size: 20px; }
        
        .notification { 
            position: fixed; 
            top: 80px; 
            left: 50%; 
            transform: translateX(-50%) translateY(-10px); 
            background: #4ecca3; 
            color: #1a1a2e; 
            padding: 12px 25px; 
            border-radius: 10px; 
            font-weight: bold; 
            opacity: 0; 
            transition: all 0.2s; 
            z-index: 101; 
        }
        .notification.show { opacity: 1; transform: translateX(-50%) translateY(0); }
        
        /* Detail popup */
        .detail-overlay { 
            display: none; 
            position: fixed; 
            top: 0; left: 0; right: 0; bottom: 0; 
            background: rgba(0,0,0,0.85); 
            z-index: 200; 
            justify-content: center; 
            align-items: center; 
        }
        .detail-overlay.show { display: flex; }
        
        .detail-box { 
            background: linear-gradient(135deg, #1e3a5f, #1a1a2e); 
            border-radius: 25px; 
            padding: 40px 50px; 
            max-width: 550px; 
            text-align: center; 
            border: 3px solid #4ecca3;
            box-shadow: 0 0 50px rgba(78,204,163,0.3);
        }
        .detail-icon { font-size: 80px; margin-bottom: 20px; }
        .detail-title { font-size: 28px; margin-bottom: 15px; color: #4ecca3; }
        .detail-info { font-size: 16px; line-height: 1.8; color: rgba(255,255,255,0.9); margin-bottom: 25px; }
        .detail-location { 
            background: rgba(78,204,163,0.2); 
            padding: 15px 25px; 
            border-radius: 10px; 
            font-size: 18px;
            margin-bottom: 25px;
        }
        .detail-close { 
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            margin: 0 auto;
            padding: 12px 30px; 
            background: rgba(255,255,255,0.1); 
            color: white; 
            border: 2px solid rgba(255,255,255,0.3); 
            border-radius: 10px; 
            font-size: 14px; 
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">City <span>Hospital</span></div>
        <div class="status">
            <div class="status-dot" id="dot"></div>
            <span id="statusTxt">Connecting...</span>
        </div>
    </div>
    
    <div class="main">
        <div class="gesture-bar">
            <div class="gesture-icon" id="gIcon">👋</div>
            <div class="gesture-name" id="gName">Show your hand</div>
        </div>
        
        <h2>How can we help you today?</h2>
        
        <div class="cards" id="cardGrid">
            <div class="card" data-idx="0">
                <div class="card-icon">👨‍⚕️</div>
                <div class="card-title">Find a Doctor</div>
                <div class="card-desc">Search specialists</div>
            </div>
            <div class="card" data-idx="1">
                <div class="card-icon">📋</div>
                <div class="card-title">Check In</div>
                <div class="card-desc">Appointments</div>
            </div>
            <div class="card" data-idx="2">
                <div class="card-icon">🚨</div>
                <div class="card-title">Emergency</div>
                <div class="card-desc">Urgent care</div>
            </div>
            <div class="card" data-idx="3">
                <div class="card-icon">🗺️</div>
                <div class="card-title">Floor Maps</div>
                <div class="card-desc">Navigate</div>
            </div>
            <div class="card" data-idx="4">
                <div class="card-icon">💊</div>
                <div class="card-title">Pharmacy</div>
                <div class="card-desc">Prescriptions</div>
            </div>
            <div class="card" data-idx="5">
                <div class="card-icon">💳</div>
                <div class="card-title">Billing</div>
                <div class="card-desc">Payments</div>
            </div>
        </div>
        
        <div class="card-nav">
            <span class="card-counter">Card <span id="cardNum">1</span> of 6</span>
        </div>
    </div>
    
    <div class="instructions">
        <div class="instruction"><span>👈</span> Swipe Left</div>
        <div class="instruction"><span>👉</span> Swipe Right</div>
        <div class="instruction"><span>👍</span> Select</div>
        <div class="instruction"><span>✊</span> Close</div>
    </div>
    
    <div class="notification" id="notif"></div>
    
    <!-- Detail popup -->
    <div class="detail-overlay" id="detailOverlay">
        <div class="detail-box">
            <div class="detail-icon" id="detailIcon">👨‍⚕️</div>
            <div class="detail-title" id="detailTitle">Find a Doctor</div>
            <div class="detail-info" id="detailInfo">Information about this service...</div>
            <div class="detail-location" id="detailLocation">📍 Floor 1, Main Lobby</div>
            <div class="detail-close">✊ Show FIST to close</div>
        </div>
    </div>
    
    <script>
        // Card data
        const cardData = [
            { icon: "👨‍⚕️", title: "Find a Doctor", info: "Search our directory of over 200 specialists across 30 departments. Find the right doctor for your needs.", location: "📍 Directory at Main Lobby" },
            { icon: "📋", title: "Check In", info: "Check in for your scheduled appointment. Please have your confirmation number or ID ready.", location: "📍 Check-in Desk, Floor 1" },
            { icon: "🚨", title: "Emergency", info: "For medical emergencies. Our ER is open 24/7 with immediate care available.", location: "📍 Emergency Room, East Wing" },
            { icon: "🗺️", title: "Floor Maps", info: "Interactive maps to help you navigate. Find departments, restrooms, cafeteria, and more.", location: "📍 Maps available at all elevators" },
            { icon: "💊", title: "Pharmacy", info: "Pick up prescriptions and get medication consultations. Open daily 7 AM - 10 PM.", location: "📍 Pharmacy, Floor 1 West Wing" },
            { icon: "💳", title: "Billing", info: "Payment services, insurance questions, and financial assistance programs.", location: "📍 Billing Office, Floor 2 Room 201" }
        ];
        
        let selectedIdx = 0;
        let detailOpen = false;
        
        const icons = {
            'NONE':'👋', 'OPEN_PALM':'✋', 'FIST':'✊', 'POINTING':'👆',
            'THUMBS_UP':'👍', 'PEACE':'✌️', 'SWIPE_LEFT':'👈', 'SWIPE_RIGHT':'👉'
        };
        
        function updateSelection() {
            document.querySelectorAll('.card').forEach((card, i) => {
                card.classList.toggle('selected', i === selectedIdx);
            });
            document.getElementById('cardNum').textContent = selectedIdx + 1;
        }
        
        function openDetail() {
            const data = cardData[selectedIdx];
            document.getElementById('detailIcon').textContent = data.icon;
            document.getElementById('detailTitle').textContent = data.title;
            document.getElementById('detailInfo').textContent = data.info;
            document.getElementById('detailLocation').textContent = data.location;
            document.getElementById('detailOverlay').classList.add('show');
            detailOpen = true;
            notify('Opened: ' + data.title);
        }
        
        function closeDetail() {
            document.getElementById('detailOverlay').classList.remove('show');
            detailOpen = false;
            notify('Closed');
        }
        
        function moveLeft() {
            if (detailOpen) return;
            selectedIdx = (selectedIdx - 1 + 6) % 6;
            updateSelection();
            notify('← ' + cardData[selectedIdx].title);
        }
        
        function moveRight() {
            if (detailOpen) return;
            selectedIdx = (selectedIdx + 1) % 6;
            updateSelection();
            notify('→ ' + cardData[selectedIdx].title);
        }
        
        function handleGesture(gesture) {
            document.getElementById('gIcon').textContent = icons[gesture] || '❓';
            document.getElementById('gName').textContent = gesture.replace('_', ' ');
            
            if (detailOpen) {
                // Only FIST closes when detail is open
                if (gesture === 'FIST') {
                    closeDetail();
                }
                return;
            }
            
            switch(gesture) {
                case 'SWIPE_LEFT':
                    moveLeft();
                    break;
                case 'SWIPE_RIGHT':
                    moveRight();
                    break;
                case 'THUMBS_UP':
                    openDetail();
                    break;
                case 'FIST':
                    // Go back to first card
                    selectedIdx = 0;
                    updateSelection();
                    notify('Reset to first');
                    break;
            }
        }
        
        function notify(text) {
            const n = document.getElementById('notif');
            n.textContent = text;
            n.classList.add('show');
            setTimeout(() => n.classList.remove('show'), 1200);
        }
        
        function connect() {
            const ws = new WebSocket('ws://localhost:8765');
            
            ws.onopen = () => {
                document.getElementById('dot').classList.add('ok');
                document.getElementById('statusTxt').textContent = 'Connected';
                notify('Connected! Swipe to navigate, Thumbs Up to select');
            };
            
            ws.onclose = () => {
                document.getElementById('dot').classList.remove('ok');
                document.getElementById('statusTxt').textContent = 'Reconnecting...';
                setTimeout(connect, 2000);
            };
            
            ws.onmessage = (e) => {
                const d = JSON.parse(e.data);
                if (d.gesture && d.gesture !== 'NONE') {
                    handleGesture(d.gesture);
                }
            };
        }
        
        // Keyboard fallback
        document.addEventListener('keydown', (e) => {
            switch(e.key) {
                case 'ArrowLeft': moveLeft(); break;
                case 'ArrowRight': moveRight(); break;
                case 'Enter': detailOpen ? closeDetail() : openDetail(); break;
                case 'Escape': closeDetail(); break;
            }
        });
        
        // Mouse click
        document.querySelectorAll('.card').forEach((card, i) => {
            card.addEventListener('click', () => {
                selectedIdx = i;
                updateSelection();
                openDetail();
            });
        });
        
        // Initialize
        updateSelection();
        connect();
    </script>
</body>
</html>"""


async def main():
    print("=" * 50)
    print("TOUCHLESS KIOSK - Day 5")
    print("=" * 50)

    server = GestureServer()

    try:
        print("\nStarting camera...")
        server.start()

        print("Starting WebSocket on ws://localhost:8765")
        await websockets.serve(server.handle_client, "localhost", 8765)

        ui_dir = Path("kiosk_ui")
        ui_dir.mkdir(exist_ok=True)
        html_path = ui_dir / "index.html"
        html_path.write_text(KIOSK_HTML, encoding="utf-8")

        print(f"Opening browser...")
        webbrowser.open(f"file:///{html_path.absolute()}")

        print("\n" + "-" * 50)
        print("CONTROLS:")
        print("  👈 Swipe LEFT   = Previous card")
        print("  👉 Swipe RIGHT  = Next card")
        print("  👍 THUMBS UP    = Open selected card")
        print("  ✊ FIST         = Close / Reset")
        print("\nKeyboard: Arrow keys, Enter, Escape")
        print("\n>>> Press 'q' in the CAMERA WINDOW to quit <<<")
        print("-" * 50 + "\n")

        await server.capture_loop()

    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        server.stop()
        print("\nKiosk stopped. Goodbye!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExited.")
