from flask import Flask, request, render_template_string, Response
import torch
from torchvision import transforms
from PIL import Image
import io
import base64
import cv2
import numpy as np

app = Flask(__name__)

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
MODEL_PATH = "perfect_coffee_model.pth" 
CONFIDENCE_THRESHOLD = 0.70

# Must remain 4 classes to prevent PyTorch from crashing, 
# but "Dark" is now excluded from the visual counts.
CLASS_NAMES = [
    "Dark",
    "Green", 
    "Red", 
    "Yellow"
]

# ==========================================
# 2. AI MODEL LOADING & PREPARATION
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.eval()
    
    num_classes = model.classifier[1].out_features
    if len(CLASS_NAMES) < num_classes:
        for i in range(len(CLASS_NAMES), num_classes):
            CLASS_NAMES.append(f"Auto-Detected Class {i}")
            
    print(f"✅ AI Model loaded successfully on {device}! Expected {num_classes} classes.")
except Exception as e:
    print(f"⚠️ Error loading AI model: {e}. The system will not function correctly without it.")
    model = None

infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 3. AI-DRIVEN BATCH & LIVE LOGIC
# ==========================================
def process_coffee_batch_bgr(img_bgr, draw_live_stats=False):
    output_img = img_bgr.copy()
    
    # 1. Isolate objects using Watershed
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
    
    red_count = 0
    yellow_count = 0
    green_count = 0
    invalid_count = 0
    min_bean_area = 400 
    
    valid_crops_tensors = []
    boxes = []
    
    # 2. Extract objects
    for label in np.unique(markers):
        if label == 0 or label == 1 or label == -1:
            continue
            
        bean_mask = np.zeros(gray.shape, dtype=np.uint8)
        bean_mask[markers == label] = 255
        contours, _ = cv2.findContours(bean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            cnt = contours[0]
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

    # 3. AI Structure & Shape Verification
    if valid_crops_tensors and model is not None:
        batch_tensor = torch.stack(valid_crops_tensors).to(device)
        
        with torch.no_grad():
            outputs = model(batch_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            max_probs, predicted_idxs = torch.max(probabilities, 1)
            
        for i in range(len(valid_crops_tensors)):
            prob = max_probs[i].item()
            pred_idx = predicted_idxs[i].item()
            x, y, w, h = boxes[i]
            
            pred_class = CLASS_NAMES[pred_idx]
            
            # Relegated "Dark" to the invalid category alongside low-confidence shapes
            if prob < CONFIDENCE_THRESHOLD or pred_class == "Dark":
                status = "Excluded"
                color = (150, 150, 150) # Grey BGR
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

            # Draw bounding box and label
            cv2.rectangle(output_img, (x, y), (x + w, y + h), color, 4)
            label_text = f"{status} {prob*100:.0f}%"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(output_img, (x, y - 20), (x + tw, y), color, -1)
            
            text_color = (0, 0, 0) if status == "Yellow" else (255, 255, 255)
            cv2.putText(output_img, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)

    # Live camera UI Panel (Dark removed)
    if draw_live_stats:
        cv2.rectangle(output_img, (10, 10), (180, 105), (0, 0, 0), -1)
        cv2.putText(output_img, f"Red: {red_count}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(output_img, f"Yellow: {yellow_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)
        cv2.putText(output_img, f"Green: {green_count}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return output_img, red_count, yellow_count, green_count, invalid_count

# ==========================================
# 4. WEBCAM STREAMING GENERATOR
# ==========================================
def generate_camera_frames():
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        blank = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(blank, "Error: Web Camera Not Found", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', blank)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            
            annotated_frame, _, _, _, _ = process_coffee_batch_bgr(frame, draw_live_stats=True)
            
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        camera.release() 

# ==========================================
# 5. PREMIUM MINIMALIST HTML/CSS/JS UI
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Analysis Center</title>
    <style>
        :root {
            --bg-color: #fafafa;
            --container-bg: #ffffff;
            --text-main: #111111;
            --text-muted: #666666;
            --accent: #000000;
            --border-light: #e0e0e0;
        }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; 
            background-color: var(--bg-color); color: var(--text-main); margin: 0; padding: 40px 20px; 
            display: flex; justify-content: center;
        }
        .container { 
            width: 100%; max-width: 600px; background: var(--container-bg); padding: 40px; 
            border-radius: 2px; box-shadow: 0 10px 30px rgba(0,0,0,0.04); border: 1px solid var(--border-light);
        }
        h1 { text-align: center; font-size: 22px; font-weight: 500; letter-spacing: 1px; margin-bottom: 30px; text-transform: uppercase; }
        .mode-selector {
            display: flex; justify-content: space-between; margin-bottom: 25px; 
            border-bottom: 1px solid var(--border-light); padding-bottom: 15px;
        }
        .mode-selector label { font-size: 14px; color: var(--text-main); cursor: pointer; display: flex; align-items: center; gap: 6px; }
        .upload-box { 
            border: 1px solid var(--border-light); padding: 30px 20px; text-align: center; 
            margin-bottom: 24px; background-color: var(--bg-color); border-radius: 2px;
        }
        .upload-box label { font-size: 14px; color: var(--text-muted); display: block; margin-bottom: 15px; }
        input[type="file"] { font-size: 14px; color: var(--text-main); }
        button { 
            width: 100%; padding: 16px; background-color: var(--accent); color: #ffffff; 
            border: none; border-radius: 2px; font-size: 14px; font-weight: 500; 
            letter-spacing: 1px; text-transform: uppercase; cursor: pointer; transition: 0.2s ease; 
        }
        button:hover { opacity: 0.8; }
        .result { margin-top: 30px; padding: 24px; border: 1px solid var(--border-light); text-align: center; border-radius: 2px; }
        .result h3 { margin-top: 0; font-size: 15px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; }
        .valid h3 { color: #111111; }
        .invalid h3 { color: #888888; }
        .result p { font-size: 14px; color: var(--text-muted); margin: 8px 0 0 0; line-height: 1.5; }
        .result strong { color: var(--text-main); font-weight: 600; }
        .preview { width: 100%; height: auto; margin-top: 24px; border-radius: 2px; border: 1px solid var(--border-light); }
        .error { color: #d32f2f; text-align: center; margin-top: 20px; font-size: 14px; }
        
        /* 3-Column Grid */
        .stat-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 15px; }
        .stat-box { padding: 15px 10px; border: 1px solid var(--border-light); border-radius: 2px; background: #fafafa; }
        
        #live-section { display: none; text-align: center; margin-top: 20px; }
        #live-feed { width: 100%; border-radius: 4px; border: 2px solid #000; margin-top: 15px; }
        .notice { font-size: 12px; color: #d32f2f; margin-top: 15px; }
    </style>
    
    <script>
        function toggleMode() {
            var mode = document.querySelector('input[name="mode"]:checked').value;
            var uploadForm = document.getElementById('upload-form-ui');
            var liveSection = document.getElementById('live-section');
            var liveFeed = document.getElementById('live-feed');
            var resultSection = document.getElementById('result-section');
            
            if (mode === 'live') {
                uploadForm.style.display = 'none';
                liveSection.style.display = 'block';
                liveFeed.src = "/video_feed"; 
                if(resultSection) resultSection.style.display = 'none';
            } else {
                uploadForm.style.display = 'block';
                liveSection.style.display = 'none';
                liveFeed.src = ""; 
                if(resultSection) resultSection.style.display = 'block';
            }
        }
        window.onload = toggleMode;
    </script>
</head>
<body>
    <div class="container">
        <h1>Analysis</h1>
        
        <form action="/" method="POST" enctype="multipart/form-data">
            <div class="mode-selector">
                <label>
                    <input type="radio" name="mode" value="batch" checked onchange="toggleMode()"> 
                    Batch Upload
                </label>
                <label>
                    <input type="radio" name="mode" value="single" onchange="toggleMode()"> 
                    Single Bean
                </label>
                <label>
                    <input type="radio" name="mode" value="live" onchange="toggleMode()"> 
                    Live Camera
                </label>
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
            <p style="color: var(--text-muted); font-size: 14px;">AI real-time analysis active. Position slot under webcam.</p>
            <img id="live-feed" src="" alt="Awaiting Camera...">
        </div>

        {% if error_message %}
            <div class="error">{{ error_message }}</div>
        {% endif %}

        {% if result %}
        <div id="result-section">
            {% if result.mode == 'single' %}
                <div class="result {% if result.is_valid %}valid{% else %}invalid{% endif %}">
                    <h3>{% if result.is_valid %}Match Confirmed{% else %}Invalid Match{% endif %}</h3>
                    {% if result.is_valid %}
                        <p>Classification: <strong>{{ result.class_name }}</strong></p>
                    {% else %}
                        <p>Object does not match reference parameters.</p>
                    {% endif %}
                    <p>Confidence: <strong>{{ result.confidence }}%</strong></p>
                </div>
            
            {% elif result.mode == 'batch' %}
                <div class="result valid">
                    <h3>AI Batch Analysis Complete</h3>
                    <p>Total Valid Beans Detected: <strong>{{ result.total }}</strong></p>
                    
                    <div class="stat-grid">
                        <div class="stat-box">
                            <h3 style="color: #d32f2f; margin-bottom: 5px;">{{ result.red }}</h3>
                            <span style="font-size: 11px; color: var(--text-muted); text-transform: uppercase;">Red</span>
                        </div>
                        <div class="stat-box">
                            <h3 style="color: #fbc02d; margin-bottom: 5px;">{{ result.yellow }}</h3>
                            <span style="font-size: 11px; color: var(--text-muted); text-transform: uppercase;">Yellow</span>
                        </div>
                        <div class="stat-box">
                            <h3 style="color: #2e7d32; margin-bottom: 5px;">{{ result.green }}</h3>
                            <span style="font-size: 11px; color: var(--text-muted); text-transform: uppercase;">Green</span>
                        </div>
                    </div>
                    
                    {% if result.invalid > 0 %}
                        <p class="notice">Note: {{ result.invalid }} object(s) failed verification (Excluded / Invalid).</p>
                    {% endif %}
                </div>
            {% endif %}

            {% if img_data %}
                <img class="preview" src="data:image/jpeg;base64,{{ img_data }}" alt="Analyzed Subject">
            {% endif %}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

# ==========================================
# 6. SERVER ROUTES & ROUTING LOGIC
# ==========================================
@app.route('/video_feed')
def video_feed():
    return Response(generate_camera_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

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
                
                # Dark count is no longer captured
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
    print("\n🌐 Server initialized. Open http://127.0.0.1:5000 in your browser.\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
