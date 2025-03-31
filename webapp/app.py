import os
import sys
import uuid
import json
import numpy as np
import nibabel as nib
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file, session
from werkzeug.utils import secure_filename
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import base64
import torch

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from src.models.segmentation import BrainSegmentationModel
from src.models.anomaly import AnomalyDetectionModel
from src.data.preprocessing import preprocess_scan
from src.inference.predict import predict_segmentation, detect_anomalies
from src.analysis.statistics import perform_statistical_analysis
from src.inference.visualization import (
    visualize_segmentation_results,
    visualize_anomaly_results,
    create_3d_render,
    create_region_volume_chart
)

app = Flask(__name__)
app.secret_key = 'brain_ai_secure_secret_key'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max upload
app.config['MODEL_PATH'] = os.path.join(os.path.dirname(__file__), '..', 'models')

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'nii', 'nii.gz'}

# Load models
segmentation_model = None
anomaly_model = None

# Clinical data cache
clinical_data_cache = {
    'ADNI_T1': None,
    'CDR': None,
    'MMSE': None,
    'GDSCALE': None,
    'ADAS_ADNI1': None,
    'NEUROBAT': None,
    'PTDEMOG': None,
    'ADAS_ADNIGO23': None
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_models():
    global segmentation_model, anomaly_model

    # Load the segmentation model
    try:
        segmentation_model = BrainSegmentationModel()
        segmentation_model.load(os.path.join(app.config['MODEL_PATH'], 'segmentation_model.pth'))
        app.logger.info("Segmentation model loaded successfully")
    except Exception as e:
        app.logger.error(f"Error loading segmentation model: {str(e)}")
        segmentation_model = None

    # Load the anomaly detection model
    try:
        anomaly_model = AnomalyDetectionModel()
        anomaly_model.load(os.path.join(app.config['MODEL_PATH'], 'anomaly_model.pth'))
        app.logger.info("Anomaly detection model loaded successfully")
    except Exception as e:
        app.logger.error(f"Error loading anomaly detection model: {str(e)}")
        anomaly_model = None


def load_clinical_data():
    """Load all clinical data files into memory"""
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical')

    for key in clinical_data_cache.keys():
        file_path = os.path.join(data_dir, f"{key}.csv")
        try:
            clinical_data_cache[key] = pd.read_csv(file_path)
            app.logger.info(f"Loaded {key} clinical data")
        except Exception as e:
            app.logger.error(f"Error loading {key} data: {str(e)}")
            clinical_data_cache[key] = None


@app.before_first_request
def initialize():
    load_models()
    load_clinical_data()


@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html',
                           segmentation_model_loaded=segmentation_model is not None,
                           anomaly_model_loaded=anomaly_model is not None)


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload"""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # If user does not select file, browser submits an empty file without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Generate a unique filename
            original_filename = secure_filename(file.filename)
            extension = original_filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{str(uuid.uuid4())}.{extension}"

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)

            # Store the filename in the session
            session['current_scan'] = {
                'filename': unique_filename,
                'original_name': original_filename,
                'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            flash('File successfully uploaded')
            return redirect(url_for('process_scan'))
        else:
            flash('Allowed file types are .nii and .nii.gz')
            return redirect(request.url)

    # GET request - show upload form
    return render_template('upload.html')


@app.route('/process', methods=['GET', 'POST'])
def process_scan():
    """Process the uploaded scan"""
    if 'current_scan' not in session:
        flash('Please upload a scan first')
        return redirect(url_for('upload_file'))

    if request.method == 'POST':
        # Get processing options
        options = {
            'perform_segmentation': 'perform_segmentation' in request.form,
            'detect_anomalies': 'detect_anomalies' in request.form,
            'generate_3d': 'generate_3d' in request.form
        }

        # Get the filepath
        filename = session['current_scan']['filename']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Process the scan
        try:
            # Load and preprocess the scan
            nifti_img = nib.load(filepath)
            preprocessed_scan = preprocess_scan(nifti_img)

            results = {}

            # Perform segmentation if requested
            if options['perform_segmentation'] and segmentation_model is not None:
                segmentation_result = predict_segmentation(preprocessed_scan, segmentation_model)

                # Save segmentation results
                results_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'results')
                os.makedirs(results_dir, exist_ok=True)

                seg_filename = f"{filename.split('.')[0]}_segmentation.nii.gz"
                seg_filepath = os.path.join(results_dir, seg_filename)

                # Save segmentation as NIfTI
                seg_img = nib.Nifti1Image(segmentation_result, nifti_img.affine)
                nib.save(seg_img, seg_filepath)

                # Generate visualization
                viz_filename = f"{filename.split('.')[0]}_segmentation.png"
                viz_filepath = os.path.join(results_dir, viz_filename)

                visualize_segmentation_results(
                    preprocessed_scan,
                    segmentation_result,
                    output_path=viz_filepath
                )

                # Add to results
                results['segmentation'] = {
                    'file': seg_filename,
                    'visualization': viz_filename
                }

                # Generate volume chart
                volume_chart_filename = f"{filename.split('.')[0]}_volumes.png"
                volume_chart_filepath = os.path.join(results_dir, volume_chart_filename)

                create_region_volume_chart(
                    segmentation_result,
                    output_path=volume_chart_filepath
                )

                results['volume_chart'] = volume_chart_filename

            # Detect anomalies if requested
            if options['detect_anomalies'] and anomaly_model is not None:
                anomaly_result, anomaly_score = detect_anomalies(preprocessed_scan, anomaly_model)

                # Save anomaly results
                results_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'results')
                os.makedirs(results_dir, exist_ok=True)

                anom_filename = f"{filename.split('.')[0]}_anomaly.nii.gz"
                anom_filepath = os.path.join(results_dir, anom_filename)

                # Save anomaly map as NIfTI
                anom_img = nib.Nifti1Image(anomaly_result, nifti_img.affine)
                nib.save(anom_img, anom_filepath)

                # Generate visualization
                anom_viz_filename = f"{filename.split('.')[0]}_anomaly.png"
                anom_viz_filepath = os.path.join(results_dir, anom_viz_filename)

                visualize_anomaly_results(
                    preprocessed_scan,
                    anomaly_result,
                    output_path=anom_viz_filepath
                )

                # Add to results
                results['anomaly'] = {
                    'file': anom_filename,
                    'visualization': anom_viz_filename,
                    'score': float(anomaly_score)
                }

            # Generate 3D model if requested
            if options['generate_3d']:
                # Create 3D rendering
                render_filename = f"{filename.split('.')[0]}_3d.html"
                render_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'results', render_filename)

                # If segmentation was performed, use it for better 3D visualization
                input_for_3d = segmentation_result if 'segmentation' in results else preprocessed_scan

                create_3d_render(
                    input_for_3d,
                    nifti_img.affine,
                    output_path=render_filepath
                )

                results['render_3d'] = render_filename

            # Store results in session
            session['processing_results'] = results

            flash('Processing complete')
            return redirect(url_for('view_results'))

        except Exception as e:
            flash(f'Error processing scan: {str(e)}')
            app.logger.error(f"Error processing scan: {str(e)}")
            return redirect(url_for('upload_file'))

    # GET request - show processing options form
    return render_template('process.html',
                           scan_info=session['current_scan'],
                           segmentation_available=segmentation_model is not None,
                           anomaly_detection_available=anomaly_model is not None)


@app.route('/results')
def view_results():
    """View processing results"""
    if 'current_scan' not in session or 'processing_results' not in session:
        flash('No processed scan available')
        return redirect(url_for('upload_file'))

    # Get the results
    results = session['processing_results']
    scan_info = session['current_scan']

    # Prepare data for template
    result_data = {
        'scan_info': scan_info,
        'results': results,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return render_template('results.html', **result_data)


@app.route('/data/<path:filename>')
def download_file(filename):
    """Download a processed file"""
    # Check if it's a result file or the original scan
    if filename.startswith('results/'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    else:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        flash(f'Error downloading file: {str(e)}')
        return redirect(url_for('view_results'))


@app.route('/analysis')
def statistical_analysis():
    """Render the statistical analysis page"""
    # Get the list of available clinical data
    available_data = {k: v is not None for k, v in clinical_data_cache.items()}

    return render_template('analysis.html', available_data=available_data)


@app.route('/api/analysis', methods=['POST'])
def perform_analysis():
    """API endpoint to perform statistical analysis"""
    # Get requested analysis parameters
    data = request.json

    analysis_type = data.get('analysis_type', 'group_comparison')
    clinical_data_keys = data.get('clinical_data', [])
    groups = data.get('groups', [])

    try:
        # Prepare input data for the analysis
        analysis_data = {}

        for key in clinical_data_keys:
            if key in clinical_data_cache and clinical_data_cache[key] is not None:
                analysis_data[key] = clinical_data_cache[key]

        # Perform the analysis
        if not analysis_data:
            return jsonify({'error': 'No clinical data available for analysis'})

        results = perform_statistical_analysis(
            analysis_data=analysis_data,
            analysis_type=analysis_type,
            groups=groups
        )

        # Return the results
        return jsonify({
            'success': True,
            'results': results
        })

    except Exception as e:
        app.logger.error(f"Error in statistical analysis: {str(e)}")
        return jsonify({'error': str(e)})


@app.route('/api/subject_data/<subject_id>')
def get_subject_data(subject_id):
    """API endpoint to get all data for a specific subject"""
    try:
        subject_data = {}

        for key, df in clinical_data_cache.items():
            if df is not None:
                # Look for subject ID column (could be named differently in each file)
                id_col = None
                for col in df.columns:
                    if 'ID' in col.upper() or 'PTID' in col.upper() or 'RID' in col.upper():
                        id_col = col
                        break

                if id_col is not None:
                    # Extract data for this subject
                    subject_rows = df[df[id_col] == subject_id]
                    if not subject_rows.empty:
                        subject_data[key] = subject_rows.to_dict(orient='records')

        return jsonify({
            'subject_id': subject_id,
            'data': subject_data
        })

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/demographic_overview')
def demographic_overview():
    """API endpoint to get demographic overview"""
    try:
        # Check if demographic data is available
        if clinical_data_cache['PTDEMOG'] is None:
            return jsonify({'error': 'Demographic data not available'})

        # Get the demographic data
        demog = clinical_data_cache['PTDEMOG']

        # Extract age distribution
        age_col = None
        for col in demog.columns:
            if 'AGE' in col.upper():
                age_col = col
                break

        age_stats = None
        if age_col:
            age_stats = {
                'mean': float(demog[age_col].mean()),
                'median': float(demog[age_col].median()),
                'min': float(demog[age_col].min()),
                'max': float(demog[age_col].max()),
                'histogram': [int(x) for x in np.histogram(demog[age_col].dropna(), bins=10)[0]]
            }

        # Extract gender distribution
        gender_col = None
        for col in demog.columns:
            if 'SEX' in col.upper() or 'GENDER' in col.upper():
                gender_col = col
                break

        gender_stats = None
        if gender_col:
            gender_counts = demog[gender_col].value_counts().to_dict()
            gender_stats = {
                'counts': gender_counts,
                'total': sum(gender_counts.values())
            }

        # Return the overview
        return jsonify({
            'total_subjects': len(demog),
            'age_stats': age_stats,
            'gender_stats': gender_stats
        })

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')


@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return render_template('500.html'), 500


@app.route('/health')
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'segmentation_model': segmentation_model is not None,
        'anomaly_model': anomaly_model is not None,
        'clinical_data': {k: v is not None for k, v in clinical_data_cache.items()}
    })


@app.route('/documentation')
def documentation():
    """Render the API documentation page"""
    return render_template('documentation.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Placeholder - Implement user registration if authentication is needed"""
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Placeholder - Implement user login if authentication is needed"""
    return render_template('login.html')


if __name__ == '__main__':
    # For development - Use a production WSGI server in production
    app.run(debug=True, host='0.0.0.0', port=5000)