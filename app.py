from flask import Flask, request, render_template_string, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
import base64
import cv2
import numpy as np
import os

# Optimize PyTorch to use fewer resources on cloud servers
torch.set_num_threads(1)

app = Flask(__name__)

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
MODEL_PATH = "perfect_coffee_model.pth" 
CONFIDENCE_THRESHOLD = 0.70

CLASS_NAMES = [
    "Dark",
    "Green", 
    "Red", 
    "Yellow"
]

# ==========================================
# 2. AI MODEL LOADING & PREPARATION
# ==========================================
device = torch.device('cpu') 
try:
    if os.path.exists(MODEL_PATH):
        model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.eval()
        num_classes = model.classifier[1].out_features
        if len(CLASS_NAMES) < num_classes:
            for i in range(len(CLASS_NAMES), num_classes):
                CLASS_NAMES.append(f"Auto-Detected Class {i}")
        print(f"✅ AI Model loaded successfully! Expected {num_classes} classes.")
    else:
        print(f"⚠️ Model file '{MODEL_PATH}' not found in directory.")
        model = None
except Exception as e:
    print(f"⚠️ Error loading AI model: {e}")
    model = None

infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 3. HYBRID BATCH LOGIC (AI + COLOR FALLBACK)
# ==========================================
def process_coffee_batch_bgr(img_bgr, draw_live_stats=False):
    output_img = img_bgr.copy()
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    cv2.watershed(img_bgr, markers)
    
    red_count, yellow_count, green_count, invalid_count = 0, 0, 0, 0
    min_bean_area = 150 # Adjusted for downscaled fast-camera feed
    
    valid_crops_tensors = []
    boxes = []
    masks = []
    
    for label in np.unique(markers):
        if label == 0 or label == 1 or label == -1:
            continue
            
        bean_mask = np.zeros(gray.shape, dtype=np.uint8)
        bean_mask[markers == label] = 255
        contours, _ = cv2.findContours(bean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            cnt = max(contours, key=cv2.contourArea) 
            area = cv2.contourArea(cnt)
            
            if area > min_bean_area:
                x, y, w, h = cv2.boundingRect(cnt)
                
                pad_x, pad_y = int(w * 0.15), int(h * 0.15)
                x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
                x2, y2 = min(img_bgr.shape[1], x + w + pad_x), min(img_bgr.shape[0], y + h + pad_y)
                
                crop_bgr = img_bgr[y1:y2, x1:x2]
                if crop_bgr.size == 0: continue
                
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(crop_rgb)
                tensor = infer_transform(pil_img)
                
                valid_crops_tensors.append(tensor)
                boxes.append((x, y, w, h))
                masks.append(bean_mask)

    if len(boxes) > 0:
        if model is not None:
            # --- AI CLASSIFICATION PATH ---
            batch_tensor = torch.stack(valid_crops_tensors).to(device)
            
            with torch.no_grad():
                outputs = model(batch_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                max_probs, predicted_idxs = torch.max(probabilities, 1)
                
            for i in range(len(boxes)):
                prob = max_probs[i].item()
                pred_idx = predicted_idxs[i].item()
                x, y, w, h = boxes[i]
                
                pred_class = CLASS_NAMES[pred_idx]
                
                if prob < CONFIDENCE_THRESHOLD or pred_class == "Dark":
                    status, color = "Excluded", (150, 150, 150)
                    invalid_count += 1
                else:
                    if pred_class == "Red":
                        status, color = "Red", (0, 0, 255)
                        red_count += 1
                    elif pred_class == "Yellow":
                        status, color = "Yellow", (0, 215, 255)
                        yellow_count += 1
                    elif pred_class == "Green":
                        status, color = "Green", (0, 255, 0)
                        green_count += 1
                    else:
                        status, color = pred_class, (200, 200, 200)

                cv2.rectangle(output_img, (x, y), (x + w, y + h), color, 4)
                label_text = f"{status} {prob*100:.0f}%"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(output_img, (x, y - 20), (x + tw, y), color, -1)
                text_color = (0, 0, 0) if status == "Yellow" else (255, 255, 255)
                cv2.putText(output_img, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)

        else:
            # --- FAILSAFE COLOR PATH ---
            lower_red1, upper_red1 = np.array([0, 50, 50]), np.array([12, 255, 255])
            lower_red2, upper_red2 = np.array([160, 50, 50]), np.array([179, 255, 255])
            lower_yellow, upper_yellow = np.array([13, 50, 50]), np.array([30, 255, 255])
            lower_green, upper_green = np.array([31, 40, 40]), np.array([95, 255, 255])
            
            for i in range(len(boxes)):
                x, y, w, h = boxes[i]
                bean_mask = masks[i]
                
                mask_r1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
                mask_r2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
                mask_red = cv2.bitwise_and(cv2.bitwise_or(mask_r1, mask_r2), bean_mask)
                mask_yellow = cv2.bitwise_and(cv2.inRange(img_hsv, lower_yellow, upper_yellow), bean_mask)
                mask_green = cv2.bitwise_and(cv2.inRange(img_hsv, lower_green, upper_green), bean_mask)
                
                count_red = cv2.countNonZero(mask_red)
                count_yellow = cv2.countNonZero(mask_yellow)
                count_green = cv2.countNonZero(mask_green)
                max_count = max(count_red, count_yellow, count_green)
                
                if max_count == count_red and count_red > 0:
                    status, color = "Red", (0, 0, 255)
                    red_count += 1
                elif max_count == count_yellow and count_yellow > 0:
                    status, color = "Yellow", (0, 215, 255)
                    yellow_count += 1
                else:
                    status, color = "Green", (0, 255, 0)
                    green_count += 1
                    
                cv2.rectangle(output_img, (x, y), (x + w, y + h), color, 4)
                label_text = f"{status} (CV Fallback)"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(output_img, (x, y - 20), (x + tw, y), color, -1)
                text_color = (0, 0, 0) if status == "Yellow" else (255, 255, 255)
                cv2.putText(output_img, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)

    if draw_live_stats:
        cv2.rectangle(output_img, (10, 10), (180, 105), (0, 0, 0), -1)
        cv2.putText(output_img, f"Red: {red_count}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(output_img, f"Yellow: {yellow_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)
        cv2.putText(output_img, f"Green: {green_count}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return output_img, red_count, yellow_count, green_count, invalid_count

# ==========================================
# 4. PREMIUM HTML/CSS/JS UI
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Analysis Center</title>
    <style>
        :root {
            --bg-color: #fafafa; --container-bg: #ffffff;
            --text-main: #111111; --text-muted: #666666;
            --accent: #000000; --border-light: #e0e0e0;
        }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg-color); color: var(--text-main); margin: 0; padding: 40px 20px; display: flex; justify-content: center; }
        .container { width: 100%; max-width: 600px; background: var(--container-bg); padding: 40px; border-radius: 2px; box-shadow: 0 10px 30px rgba(0,0,0,0.04); border: 1px solid var(--border-light); }
        h1 { text-align: center; font-size: 22px; font-weight: 500; letter-spacing: 1px; margin-bottom: 30px; text-transform: uppercase; }
        .mode-selector { display: flex; justify-content: space-between; margin-bottom: 25px; border-bottom: 1px solid var(--border-light); padding-bottom: 15px; }
        .mode-selector label { font-size: 14px; cursor: pointer; display: flex; align-items: center; gap: 6px; }
        .upload-box { border: 1px solid var(--border-light); padding: 30px 20px; text-align: center; margin-bottom: 24px; background: var(--bg-color); border-radius: 2px; }
        .upload-box label { font-size: 14px; color: var(--text-muted); display: block; margin-bottom: 15px; }
        input[type="file"] { font-size: 14px; }
        button { width: 100%; padding: 16px; background: var(--accent); color: #fff; border: none; font-size: 14px; font-weight: 500; letter-spacing: 1px; text-transform: uppercase; cursor: pointer; transition: 0.2s; }
        button:hover { opacity: 0.8; }
        .result { margin-top: 30px; padding: 24px; border: 1px solid var(--border-light); text-align: center; }
        .result h3 { margin-top: 0; font-size: 15px; text-transform: uppercase; }
        .result p { font-size: 14px; color: var(--text-muted); }
        .preview { width: 100%; margin-top: 24px; border: 1px solid var(--border-light); }
        .error { color: #d32f2f; text-align: center; margin-top: 20px; font-size: 14px; }
        .stat-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 15px; }
        .stat-box { padding: 15px 10px; border: 1px solid var(--border-light); background: #fafafa; text-align: center;}
        
        #live-section { display: none; text-align: center; margin-top: 20px; }
        #client-video { display: none; } 
        #live-feed { width: 100%; border-radius: 4px; border: 1px solid var(--border-light); margin-top: 15px; min-height: 300px; background: #eee;}
        .notice { font-size: 12px; color: #d32f2f; margin-top: 15px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Analysis</h1>
        
        <form action="/" method="POST" enctype="multipart/form-data" id="main-form">
            <div class="mode-selector">
                <label><input type="radio" name="mode" value="batch" checked onchange="toggleMode()"> Batch Upload</label>
                <label><input type="radio" name="mode" value="single" onchange="toggleMode()"> Single Bean</label>
                <label><input type="radio" name="mode" value="live" onchange="toggleMode()"> Live Camera</label>
            </div>

            <div id="upload-form-ui">
                <div class="upload-box">
                    <label>Select an image to analyze</label>
                    <input type="file" name="file" accept=".jpg, .jpeg, .png">
                </div>
                <button type="submit">Process Image</button>
            </div>
        </form>

        <div id="live-section">
            <p style="color: var(--text-muted); font-size: 14px;">Live Processing Active. Position slot under webcam.</p>
            <video id="client-video" autoplay playsinline></video>
            <canvas id="client-canvas" style="display:none;"></canvas>
            <img id="live-feed" src="" alt="Requesting Camera Access...">
            
            <div class="stat-grid" id="live-stats" style="display:none;">
                <div class="stat-box"><h3 style="color: #d32f2f; margin:0;" id="st-red">0</h3><span style="font-size: 11px; color: var(--text-muted); text-transform: uppercase;">Red</span></div>
                <div class="stat-box"><h3 style="color: #fbc02d; margin:0;" id="st-yel">0</h3><span style="font-size: 11px; color: var(--text-muted); text-transform: uppercase;">Yellow</span></div>
                <div class="stat-box"><h3 style="color: #2e7d32; margin:0;" id="st-grn">0</h3><span style="font-size: 11px; color: var(--text-muted); text-transform: uppercase;">Green</span></div>
            </div>
        </div>

        {% if error_message %}
            <div class="error">{{ error_message }}</div>
        {% endif %}

        {% if result %}
        <div id="result-section">
            {% if result.mode == 'single' %}
                <div class="result {% if result.is_valid %}valid{% else %}invalid{% endif %}">
                    <h3>{% if result.is_valid %}Match Confirmed{% else %}Invalid Match{% endif %}</h3>
                    <p>Classification: <strong>{{ result.class_name }}</strong></p>
                    <p>Confidence: <strong>{{ result.confidence }}%</strong></p>
                </div>
            {% elif result.mode == 'batch' %}
                <div class="result valid">
                    <h3>Batch Analysis Complete</h3>
                    <p>Total Valid Beans Detected: <strong>{{ result.total }}</strong></p>
                    <div class="stat-grid">
                        <div class="stat-box"><h3 style="color: #d32f2f; margin-bottom: 5px;">{{ result.red }}</h3><span style="font-size: 11px; color: var(--text-muted); text-transform: uppercase;">Red</span></div>
                        <div class="stat-box"><h3 style="color: #fbc02d; margin-bottom: 5px;">{{ result.yellow }}</h3><span style="font-size: 11px; color: var(--text-muted); text-transform: uppercase;">Yellow</span></div>
                        <div class="stat-box"><h3 style="color: #2e7d32; margin-bottom: 5px;">{{ result.green }}</h3><span style="font-size: 11px; color: var(--text-muted); text-transform: uppercase;">Green</span></div>
                    </div>
                </div>
            {% endif %}
            {% if img_data %}
                <img class="preview" src="data:image/jpeg;base64,{{ img_data }}" alt="Analyzed Subject">
            {% endif %}
        </div>
        {% endif %}
    </div>

    <!-- Optimized Client-Side Camera Logic -->
    <script>
        let stream = null;
        let isLiveMode = false;
        let isProcessing = false;
        const video = document.getElementById('client-video');
        const canvas = document.getElementById('client-canvas');
        const ctx = canvas.getContext('2d');
        const liveFeed = document.getElementById('live-feed');
        const liveStats = document.getElementById('live-stats');

        function toggleMode() {
            var mode = document.querySelector('input[name="mode"]:checked').value;
            if (mode === 'live') {
                document.getElementById('upload-form-ui').style.display = 'none';
                document.getElementById('live-section').style.display = 'block';
                if(document.getElementById('result-section')) document.getElementById('result-section').style.display = 'none';
                isLiveMode = true;
                startCamera();
            } else {
                document.getElementById('upload-form-ui').style.display = 'block';
                document.getElementById('live-section').style.display = 'none';
                if(document.getElementById('result-section')) document.getElementById('result-section').style.display = 'block';
                isLiveMode = false;
                stopCamera();
            }
        }

        async function startCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { facingMode: 'environment', width: { ideal: 640 } } 
                });
                video.srcObject = stream;
                liveStats.style.display = 'grid';
                
                video.onplaying = () => {
                    requestAnimationFrame(processFrameLoop);
                };
            } catch (err) {
                alert("Camera access denied or unavailable.");
            }
        }

        function stopCamera() {
            if (stream) { stream.getTracks().forEach(track => track.stop()); }
        }

        // Smart Async Loop: Won't send a new frame until the server replies to the previous one
        async function processFrameLoop() {
            if (!isLiveMode) return;

            if (!isProcessing && video.readyState === video.HAVE_ENOUGH_DATA) {
                isProcessing = true;
                
                // Downscale frame to a max width of 640px to eliminate network lag
                const MAX_WIDTH = 640;
                const scale = Math.min(MAX_WIDTH / video.videoWidth, 1.0);
                canvas.width = video.videoWidth * scale;
                canvas.height = video.videoHeight * scale;
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                
                // Compress JPEG heavily to send bytes faster
                const dataURL = canvas.toDataURL('image/jpeg', 0.5); 
                const formData = new FormData();
                formData.append('frame', dataURL);

                try {
                    const response = await fetch('/live_frame', { method: 'POST', body: formData });
                    const result = await response.json();
                    
                    if (result.image) {
                        liveFeed.src = 'data:image/jpeg;base64,' + result.image;
                        document.getElementById('st-red').innerText = result.red;
                        document.getElementById('st-yel').innerText = result.yellow;
                        document.getElementById('st-grn').innerText = result.green;
                    }
                } catch (e) {
                    console.error("Frame dropped:", e);
                }
                
                isProcessing = false;
            }
            
            requestAnimationFrame(processFrameLoop);
        }

        window.onload = toggleMode;
    </script>
</body>
</html>
"""

# ==========================================
# 5. SERVER ROUTES & ROUTING LOGIC
# ==========================================

@app.route('/live_frame', methods=['POST'])
def live_frame():
    data_url = request.form.get('frame')
    if not data_url:
        return jsonify({'error': 'No data'})
        
    try:
        header, encoded = data_url.split(",", 1)
        data = base64.b64decode(encoded)
        nparr = np.frombuffer(data, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        output_img_bgr, red, yellow, green, invalid = process_coffee_batch_bgr(img_bgr, draw_live_stats=False)
        
        _, buffer = cv2.imencode('.jpg', output_img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'image': img_b64,
            'red': red,
            'yellow': yellow,
            'green': green
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        mode = request.form.get('mode', 'batch')
        
        if 'file' not in request.files or request.files['file'].filename == '':
            return render_template_string(HTML_TEMPLATE, error_message="No image provided for upload processing.")
        
        file = request.files['file']

        try:
            img_bytes = file.read()

            if mode == 'single':
                image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                if model is None:
                    return render_template_string(HTML_TEMPLATE, error_message="System offline: Model failed to load.")

                input_tensor = infer_transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    max_prob, predicted_idx = torch.max(probabilities, 0)
                
                prob_val = max_prob.item()
                predicted_class = CLASS_NAMES[predicted_idx.item()]
                is_valid = prob_val >= CONFIDENCE_THRESHOLD

                result = {
                    "mode": "single",
                    "is_valid": is_valid,
                    "class_name": predicted_class,
                    "confidence": f"{prob_val * 100:.2f}"
                }
                return render_template_string(HTML_TEMPLATE, result=result, img_data=img_b64)

            elif mode == 'batch':
                nparr = np.frombuffer(img_bytes, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                output_img_bgr, red, yellow, green, invalid = process_coffee_batch_bgr(img_bgr, draw_live_stats=False)
                
                output_img_rgb = cv2.cvtColor(output_img_bgr, cv2.COLOR_BGR2RGB)
                final_pil_img = Image.fromarray(output_img_rgb)
                
                buffered = io.BytesIO()
                final_pil_img.save(buffered, format="JPEG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                result = {
                    "mode": "batch",
                    "total": red + yellow + green,
                    "red": red,
                    "yellow": yellow,
                    "green": green,
                    "invalid": invalid
                }
                
                return render_template_string(HTML_TEMPLATE, result=result, img_data=img_b64)

        except Exception as e:
            return render_template_string(HTML_TEMPLATE, error_message=f"Processing error: {str(e)}")

    return render_template_string(HTML_TEMPLATE)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
