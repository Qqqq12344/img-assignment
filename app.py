import os
import cv2
import numpy as np
from PIL import Image
import json
import time
from werkzeug.utils import secure_filename
# Import extended YOLO classes
from detection_algorithms import YOLOv8Detector, YOLOv10Detector, YOLOv11Detector, YOLOv12Detector, FasterRCNN, EfficientDet
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory 
from violation_checker import ZoneAnalyzer


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize detection models
try:
    zone_analyzer = ZoneAnalyzer()
    yolo_v8 = YOLOv8Detector(model_size="n", confidence_threshold=0.5)
    yolo_v10 = YOLOv10Detector(model_size="n", confidence_threshold=0.5)
    yolo_v11 = YOLOv11Detector(model_size="n", confidence_threshold=0.5)
    yolo_v12 = YOLOv12Detector(model_size="n", confidence_threshold=0.5)
    
    faster_rcnn = FasterRCNN()
    efficient_det = EfficientDet()
    logger.info("All models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    yolo_v8 = yolo_v10 = yolo_v11 = yolo_v12 = None
    faster_rcnn = efficient_det = zone_analyzer = None

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image_security(file_path):
    """Validate uploaded image for security"""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format. Please upload an image.'}), 400
        
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file.save(file_path)
        
        if not validate_image_security(file_path):
            os.remove(file_path)
            return jsonify({'error': 'Invalid image file'}), 400
        
        return jsonify({
            'success': True,
            'filename': filename,
            'message': 'File uploaded successfully'
        })
    
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': 'Upload failed'}), 500

@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        data = request.get_json()
        if not data or 'filename' not in data:
            return jsonify({'error': 'Missing filename'}), 400
        
        filename = data['filename']
        zones_raw = data.get('zones', [])

        # After reading image, weâ€™ll get actual image size:
        display_w = int(data.get("display_width") or 1)
        display_h = int(data.get("display_height") or 1)

        # Will be calculated AFTER image is loaded below
        scale_x = scale_y = 1.0

        
        if not secure_filename(filename) == filename:
            return jsonify({'error': 'Invalid filename'}), 400
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        image = cv2.imread(file_path)
        if image is None:
            return jsonify({'error': 'Cannot read image file'}), 400
        
        height, width = image.shape[:2]

        if display_w > 1 and display_h > 1:
            display_w = data.get("displayWidth", width)
            display_h = data.get("displayHeight", height)

            scale_x = width / display_w
            scale_y = height / display_h


        def _scale_zone(z):
            scaled = dict(z)
            scaled["points"] = [[float(p[0]) * scale_x, float(p[1]) * scale_y] for p in z.get("points", [])]
            return scaled

        zones = [_scale_zone(z) for z in zones_raw]


        
        results = {
            'image_info': {
                'width': width,
                'height': height,
                'filename': filename
            },
            'detections': {},
            'violations': [],
            'performance': {},
            'occupancy': 0.0
        }
        
        # Add all YOLO versions + optional models
        algorithms = [
            ('yolov8', yolo_v8, "YOLOv8"),
            ('yolov10', yolo_v10, "YOLOv10"),
            ('yolov11', yolo_v11, "YOLOv11"),
            ('yolov12', yolo_v12, "YOLOv12"),
            ('faster_rcnn', faster_rcnn, "Faster R-CNN"),
            ('efficient_det', efficient_det, "EfficientDet")
        ]
        
        for algo_name, detector, display_name in algorithms:
            if detector is None:
                results['detections'][algo_name] = {
                    'error': 'Model not loaded',
                    'detections': [],
                    'processing_time': 0,
                    'display_name': display_name
                }
                continue
            
            try:
                start_time = time.time()
                detections = detector.detect(image)
                processing_time = time.time() - start_time
                
                results['detections'][algo_name] = {
                    'detections': detections,
                    'processing_time': round(processing_time, 3),
                    'vehicle_count': len(detections),
                    'display_name': display_name
                }
                
                logger.info(f"{display_name}: {len(detections)} vehicles detected in {processing_time:.3f}s")
            
            except Exception as e:
                logger.error(f"Error in {algo_name}: {e}")
                results['detections'][algo_name] = {
                    'error': str(e),
                    'detections': [],
                    'processing_time': 0,
                    'display_name': display_name
                }
        
        # Get selected model from frontend (default = yolov8)
        selected_algo = data.get('selected_algo', 'yolov8')

        # Use the chosen model’s detections as primary
        primary_detections = results['detections'].get(selected_algo, {}).get('detections', [])

        if zones and primary_detections:
            try:
                violations = zone_analyzer.detect_violations(primary_detections, zones)
                results['violations'] = violations
                logger.info(f"Found {len(violations)} violations using {selected_algo}")
            except Exception as e:
                logger.error(f"Violation detection error: {e}")
                results['violations'] = []

                
        total_spots = sum(1 for zone in zones if zone.get('type') in ['standard', 'vip', 'handicap'])
        if total_spots > 0:
            occupied_spots = min(len(primary_detections), total_spots)
            results['occupancy'] = round((occupied_spots / total_spots) * 100, 1)
        
        processing_times = [
            det.get('processing_time', 0) 
            for det in results['detections'].values() 
            if 'processing_time' in det
        ]
        
        results['performance'] = {
            'total_processing_time': round(sum(processing_times), 3),
            'average_processing_time': round(sum(processing_times) / len(processing_times), 3) if processing_times else 0,
            'fastest_algorithm': min(results['detections'].items(), key=lambda x: x[1].get('processing_time', float('inf')))[0] if processing_times else None
        }
        
         # Echo back selected model
        results['primary_algo'] = selected_algo


        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': 'Analysis failed'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.errorhandler(400)
def bad_request(error):
    return jsonify(error="Bad request - please check your input"), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify(error="Resource not found"), 404

@app.errorhandler(413)
def too_large(error):
    return jsonify(error="File too large. Maximum size is 10MB"), 413

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify(error="Internal server error"), 500

@app.after_request
def add_cache_header(response):
    if request.endpoint == 'uploaded_file':
        response.cache_control.max_age = 300
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
