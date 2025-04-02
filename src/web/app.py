"""
Web application module for Brain MRI Analysis Platform.
This module provides a web interface for the brain segmentation and analysis framework.
"""

import os
import sys
import logging
import nibabel as nib
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify, session
from werkzeug.utils import secure_filename
import plotly
import plotly.express as px
import json
import uuid
import io
import zipfile
from pathlib import Path
import shutil
import threading
import queue
from PIL import Image

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Import other modules from the package
from models.segmentation import BrainSegmentationModel
from models.abnormality import AbnormalityDetector
from analysis.volumetric import VolumetricAnalyzer
from analysis.clinical import ClinicalAnalyzer
from analysis.visualization import BrainVisualization

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('web_app')

# Initialize Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'brain-ai-platform-secret-key'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'results')
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 1000 MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'nii', 'nii.gz', 'mgz', 'mgh'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Create a queue for background tasks
task_queue = queue.Queue()

# Dictionary to store task status
tasks = {}

# Initialize models
segmentation_model = None
abnormality_detector = None

# Metadata
APP_VERSION = "1.0.0"
APP_CREATION_DATE = "2025-04-02 15:07:45"
APP_AUTHOR = "KishoreKumarKalli"


def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_nifti_file(file_path):
    """Load a NIfTI file and return the image data and object."""
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        return data, img
    except Exception as e:
        logger.error(f"Error loading NIfTI file: {str(e)}")
        return None, None


def process_task():
    """Background worker to process tasks from the queue."""
    while True:
        try:
            # Get task from queue
            task_id, task_type, args = task_queue.get()

            # Update task status
            tasks[task_id]['status'] = 'processing'

            # Process task based on type
            if task_type == 'segmentation':
                file_path, output_dir = args
                try:
                    run_segmentation(file_path, output_dir, task_id)
                    tasks[task_id]['status'] = 'completed'
                except Exception as e:
                    logger.error(f"Error in segmentation task {task_id}: {str(e)}")
                    tasks[task_id]['status'] = 'failed'
                    tasks[task_id]['error'] = str(e)

            elif task_type == 'abnormality':
                file_path, output_dir = args
                try:
                    run_abnormality_detection(file_path, output_dir, task_id)
                    tasks[task_id]['status'] = 'completed'
                except Exception as e:
                    logger.error(f"Error in abnormality task {task_id}: {str(e)}")
                    tasks[task_id]['status'] = 'failed'
                    tasks[task_id]['error'] = str(e)

            elif task_type == 'analysis':
                seg_path, img_path, output_dir = args
                try:
                    run_volumetric_analysis(seg_path, img_path, output_dir, task_id)
                    tasks[task_id]['status'] = 'completed'
                except Exception as e:
                    logger.error(f"Error in analysis task {task_id}: {str(e)}")
                    tasks[task_id]['status'] = 'failed'
                    tasks[task_id]['error'] = str(e)

            # Mark task as complete in queue
            task_queue.task_done()

        except Exception as e:
            logger.error(f"Error in worker thread: {str(e)}")
            # Don't terminate the thread on error
            continue


def run_segmentation(file_path, output_dir, task_id):
    """Run segmentation on the given file."""
    logger.info(f"Running segmentation on {file_path}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load the model if not already loaded
    global segmentation_model
    if segmentation_model is None:
        model_path = os.path.join(parent_dir, 'models', 'weights', 'segmentation_model.pth')
        if os.path.exists(model_path):
            segmentation_model = BrainSegmentationModel(model_path=model_path)
        else:
            segmentation_model = BrainSegmentationModel()
            logger.warning("No pre-trained segmentation model found. Using untrained model.")

    # Load the input file
    data, img = load_nifti_file(file_path)
    if data is None or img is None:
        raise ValueError(f"Failed to load input file: {file_path}")

    # Update task progress
    tasks[task_id]['progress'] = 25

    # Preprocess and normalize data if needed
    # (This step might be done inside the model's predict method)

    # Run segmentation
    seg_result = segmentation_model.predict(data, img)

    # Update task progress
    tasks[task_id]['progress'] = 75

    # Save result
    output_file = os.path.join(output_dir, 'segmentation_result.nii.gz')
    segmentation_model.save_segmentation(seg_result, img, output_file)

    # Create visualization
    visualizer = BrainVisualization(output_dir=output_dir)

    # Create multi-view visualization
    viz_file = os.path.join(output_dir, 'segmentation_visualization.png')
    visualizer.create_multiview_visualization(data, seg_result, output_path=viz_file)

    # Update task progress
    tasks[task_id]['progress'] = 100

    # Add results to task
    tasks[task_id]['results'] = {
        'segmentation_file': output_file,
        'visualization_file': viz_file
    }

    logger.info(f"Segmentation completed for task {task_id}")
    return output_file


def run_abnormality_detection(file_path, output_dir, task_id):
    """Run abnormality detection on the given file."""
    logger.info(f"Running abnormality detection on {file_path}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load the model if not already loaded
    global abnormality_detector
    if abnormality_detector is None:
        model_path = os.path.join(parent_dir, 'models', 'weights', 'abnormality_model.pth')
        if os.path.exists(model_path):
            abnormality_detector = AbnormalityDetector(model_path=model_path)
        else:
            abnormality_detector = AbnormalityDetector(method='statistical')
            logger.warning("No pre-trained abnormality model found. Using statistical method only.")

    # Load the input file
    data, img = load_nifti_file(file_path)
    if data is None or img is None:
        raise ValueError(f"Failed to load input file: {file_path}")

    # Update task progress
    tasks[task_id]['progress'] = 25

    # Run abnormality detection
    detection_result = abnormality_detector.detect(data)

    # Update task progress
    tasks[task_id]['progress'] = 75

    # Save abnormality map
    output_file = os.path.join(output_dir, 'abnormality_map.nii.gz')
    abnormality_detector.save_abnormality_map(detection_result['abnormality_map'], img, output_file)

    # Create visualization
    visualizer = BrainVisualization(output_dir=output_dir)

    # Create visualization with abnormality overlay
    viz_file = os.path.join(output_dir, 'abnormality_visualization.png')
    visualizer.create_multiview_visualization(data, abnormality_map=detection_result['abnormality_map'],
                                              output_path=viz_file)

    # Update task progress
    tasks[task_id]['progress'] = 100

    # Add results to task
    tasks[task_id]['results'] = {
        'abnormality_file': output_file,
        'visualization_file': viz_file,
        'is_abnormal': detection_result['is_abnormal'],
        'abnormality_score': float(detection_result['abnormality_score'])
    }

    logger.info(f"Abnormality detection completed for task {task_id}")
    return output_file


def run_volumetric_analysis(segmentation_path, original_img_path, output_dir, task_id):
    """Run volumetric analysis on the given segmentation file."""
    logger.info(f"Running volumetric analysis on {segmentation_path}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load the segmentation file
    seg_data, seg_img = load_nifti_file(segmentation_path)
    if seg_data is None or seg_img is None:
        raise ValueError(f"Failed to load segmentation file: {segmentation_path}")

    # Load the original image for reference
    orig_data, orig_img = load_nifti_file(original_img_path)
    if orig_data is None or orig_img is None:
        # Use segmentation image if original not available
        orig_data, orig_img = seg_data, seg_img

    # Update task progress
    tasks[task_id]['progress'] = 25

    # Initialize the volumetric analyzer
    analyzer = VolumetricAnalyzer(output_dir=output_dir)

    # Calculate volumes
    volumes_csv = os.path.join(output_dir, 'structure_volumes.csv')
    volume_results = analyzer.calculate_volumes(seg_data, seg_img, output_path=volumes_csv)

    # Update task progress
    tasks[task_id]['progress'] = 50

    # Calculate relative volumes
    rel_volumes_csv = os.path.join(output_dir, 'relative_volumes.csv')
    rel_volumes = analyzer.calculate_relative_volumes(volume_results, output_path=rel_volumes_csv)

    # Update task progress
    tasks[task_id]['progress'] = 75

    # Create visualization
    visualizer = BrainVisualization(output_dir=output_dir)

    # Create 3D rendering of selected structures
    structures = [1, 2, 3, 4]  # Grey matter, white matter, CSF, cerebellum
    render_file = os.path.join(output_dir, 'volume_3d_render.html')
    visualizer.visualize_3d_render(seg_data, structures=structures, output_path=render_file)

    # Create volume chart
    chart_file = os.path.join(output_dir, 'volume_chart.png')
    plt_fig = visualizer.create_volume_comparison_chart(
        rel_volumes,
        structures=rel_volumes['structure'].unique()[:5],  # First 5 structures
        output_path=chart_file
    )

    # Update task progress
    tasks[task_id]['progress'] = 100

    # Add results to task
    tasks[task_id]['results'] = {
        'volumes_csv': volumes_csv,
        'relative_volumes_csv': rel_volumes_csv,
        'render_file': render_file,
        'chart_file': chart_file,
        'total_brain_volume': float(volume_results['total_brain_volume']),
        'intracranial_volume': float(volume_results['intracranial_volume'])
    }

    logger.info(f"Volumetric analysis completed for task {task_id}")
    return volumes_csv


# Start background worker thread
worker_thread = threading.Thread(target=process_task, daemon=True)
worker_thread.start()


@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html',
                           version=APP_VERSION,
                           creation_date=APP_CREATION_DATE,
                           author=APP_AUTHOR)


@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html',
                           version=APP_VERSION,
                           creation_date="2025-04-02 15:08:59",
                           author="KishoreKumarKalli")


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload and processing."""
    if request.method == 'POST':
        # Check if file part exists
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # Check if file is selected
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)

        # Check if file type is allowed
        if file and allowed_file(file.filename):
            # Generate unique ID for this session/task
            session_id = str(uuid.uuid4())

            # Save the file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
            file.save(file_path)

            # Store file path in session
            session['file_path'] = file_path
            session['filename'] = filename
            session['session_id'] = session_id

            # Redirect to processing options
            return redirect(url_for('processing_options'))
        else:
            flash(f'Invalid file type. Allowed types: {", ".join(app.config["ALLOWED_EXTENSIONS"])}')
            return redirect(request.url)

    # GET request - render upload form
    return render_template('upload.html')


@app.route('/processing-options', methods=['GET'])
def processing_options():
    """Show processing options for the uploaded file."""
    if 'file_path' not in session:
        flash('No file uploaded. Please upload a file first.')
        return redirect(url_for('upload_file'))

    file_path = session.get('file_path')
    filename = session.get('filename')

    # Try to load and show a preview of the uploaded file
    data, img = load_nifti_file(file_path)
    preview_path = None

    if data is not None:
        # Generate a preview image (middle slice)
        try:
            middle_slice = data[:, :, data.shape[2] // 2]

            # Normalize to 0-255 for visualization
            middle_slice = ((middle_slice - middle_slice.min()) /
                            (middle_slice.max() - middle_slice.min()) * 255).astype(np.uint8)

            # Save as an image
            preview_path = os.path.join(app.config['UPLOAD_FOLDER'], f"preview_{session.get('session_id')}.png")
            Image.fromarray(middle_slice).save(preview_path)

            # Make path relative to static
            preview_path = os.path.relpath(preview_path, os.path.join(os.path.dirname(__file__), 'static'))
        except Exception as e:
            logger.error(f"Error generating preview: {str(e)}")
            preview_path = None

    return render_template('processing_options.html',
                           filename=filename,
                           preview_path=preview_path)


@app.route('/run-segmentation', methods=['POST'])
def run_segmentation_task():
    """Run segmentation on the uploaded file."""
    if 'file_path' not in session:
        return jsonify({'error': 'No file uploaded'}), 400

    file_path = session.get('file_path')
    session_id = session.get('session_id')

    # Create output directory
    output_dir = os.path.join(app.config['RESULTS_FOLDER'], session_id, 'segmentation')
    os.makedirs(output_dir, exist_ok=True)

    # Create task
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        'type': 'segmentation',
        'status': 'queued',
        'progress': 0,
        'session_id': session_id,
        'created_at': "2025-04-02 15:08:59",
        'created_by': "KishoreKumarKalli"
    }

    # Add task to queue
    task_queue.put((task_id, 'segmentation', (file_path, output_dir)))

    return jsonify({
        'task_id': task_id,
        'status': 'queued',
        'message': 'Segmentation task queued'
    })


@app.route('/run-abnormality', methods=['POST'])
def run_abnormality_task():
    """Run abnormality detection on the uploaded file."""
    if 'file_path' not in session:
        return jsonify({'error': 'No file uploaded'}), 400

    file_path = session.get('file_path')
    session_id = session.get('session_id')

    # Create output directory
    output_dir = os.path.join(app.config['RESULTS_FOLDER'], session_id, 'abnormality')
    os.makedirs(output_dir, exist_ok=True)

    # Create task
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        'type': 'abnormality',
        'status': 'queued',
        'progress': 0,
        'session_id': session_id,
        'created_at': "2025-04-02 15:08:59",
        'created_by': "KishoreKumarKalli"
    }

    # Add task to queue
    task_queue.put((task_id, 'abnormality', (file_path, output_dir)))

    return jsonify({
        'task_id': task_id,
        'status': 'queued',
        'message': 'Abnormality detection task queued'
    })


@app.route('/run-analysis', methods=['POST'])
def run_analysis_task():
    """Run volumetric analysis on segmentation results."""
    if 'file_path' not in session:
        return jsonify({'error': 'No file uploaded'}), 400

    session_id = session.get('session_id')
    file_path = session.get('file_path')

    # Check if segmentation results exist
    seg_path = os.path.join(app.config['RESULTS_FOLDER'], session_id, 'segmentation', 'segmentation_result.nii.gz')
    if not os.path.exists(seg_path):
        return jsonify({
            'error': 'Segmentation results not found. Run segmentation first.'
        }), 400

    # Create output directory
    output_dir = os.path.join(app.config['RESULTS_FOLDER'], session_id, 'analysis')
    os.makedirs(output_dir, exist_ok=True)

    # Create task
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        'type': 'analysis',
        'status': 'queued',
        'progress': 0,
        'session_id': session_id,
        'created_at': "2025-04-02 15:08:59",
        'created_by': "KishoreKumarKalli"
    }

    # Add task to queue
    task_queue.put((task_id, 'analysis', (seg_path, file_path, output_dir)))

    return jsonify({
        'task_id': task_id,
        'status': 'queued',
        'message': 'Analysis task queued'
    })


@app.route('/task-status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """Get status of a task."""
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404

    task = tasks[task_id]
    response = {
        'task_id': task_id,
        'status': task['status'],
        'progress': task.get('progress', 0),
        'type': task['type']
    }

    # Include results if task is completed
    if task['status'] == 'completed' and 'results' in task:
        response['results'] = task['results']

    # Include error if task failed
    if task['status'] == 'failed' and 'error' in task:
        response['error'] = task['error']

    return jsonify(response)


@app.route('/results/<session_id>', methods=['GET'])
def show_results(session_id):
    """Show results for a session."""
    # Check if results exist
    session_dir = os.path.join(app.config['RESULTS_FOLDER'], session_id)
    if not os.path.exists(session_dir):
        flash('No results found for this session')
        return redirect(url_for('index'))

    # Collect all results
    results = {
        'session_id': session_id,
        'segmentation': None,
        'abnormality': None,
        'analysis': None
    }

    # Check for segmentation results
    seg_dir = os.path.join(session_dir, 'segmentation')
    if os.path.exists(seg_dir):
        viz_file = os.path.join(seg_dir, 'segmentation_visualization.png')
        if os.path.exists(viz_file):
            results['segmentation'] = {
                'visualization': os.path.relpath(viz_file, os.path.join(os.path.dirname(__file__), 'static')),
                'download_path': f"/download/{session_id}/segmentation"
            }

    # Check for abnormality results
    abnorm_dir = os.path.join(session_dir, 'abnormality')
    if os.path.exists(abnorm_dir):
        viz_file = os.path.join(abnorm_dir, 'abnormality_visualization.png')
        if os.path.exists(viz_file):
            # Check if abnormality results include score
            score_file = os.path.join(abnorm_dir, 'abnormality_results.json')
            is_abnormal = None
            abnormality_score = None

            if os.path.exists(score_file):
                try:
                    with open(score_file, 'r') as f:
                        abnorm_data = json.load(f)
                        is_abnormal = abnorm_data.get('is_abnormal')
                        abnormality_score = abnorm_data.get('abnormality_score')
                except Exception as e:
                    logger.error(f"Error loading abnormality results: {str(e)}")

            results['abnormality'] = {
                'visualization': os.path.relpath(viz_file, os.path.join(os.path.dirname(__file__), 'static')),
                'download_path': f"/download/{session_id}/abnormality",
                'is_abnormal': is_abnormal,
                'abnormality_score': abnormality_score
            }

    # Check for analysis results
    analysis_dir = os.path.join(session_dir, 'analysis')
    if os.path.exists(analysis_dir):
        chart_file = os.path.join(analysis_dir, 'volume_chart.png')
        volumes_csv = os.path.join(analysis_dir, 'structure_volumes.csv')

        if os.path.exists(chart_file) and os.path.exists(volumes_csv):
            # Load volumes data for display
            try:
                volumes_df = pd.read_csv(volumes_csv)
                volumes_table = volumes_df.to_html(classes='table table-striped', index=False)
            except Exception as e:
                logger.error(f"Error loading volumes data: {str(e)}")
                volumes_table = None

            results['analysis'] = {
                'chart': os.path.relpath(chart_file, os.path.join(os.path.dirname(__file__), 'static')),
                'download_path': f"/download/{session_id}/analysis",
                'volumes_table': volumes_table
            }

    return render_template('results.html', results=results)


@app.route('/download/<session_id>/<result_type>', methods=['GET'])
def download_results(session_id, result_type):
    """Download results as a zip file."""
    # Check if results exist
    result_dir = os.path.join(app.config['RESULTS_FOLDER'], session_id, result_type)
    if not os.path.exists(result_dir):
        flash(f'No {result_type} results found for this session')
        return redirect(url_for('index'))

    # Create a zip file in memory
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        for root, dirs, files in os.walk(result_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(result_dir))
                zf.write(file_path, arcname)

    # Reset file pointer
    memory_file.seek(0)

    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'brain_ai_{result_type}_results_{session_id}.zip'
    )


@app.route('/documentation')
def documentation():
    """Show documentation page."""
    return render_template('documentation.html',
                           version=APP_VERSION,
                           creation_date="2025-04-02 15:08:59",
                           author="KishoreKumarKalli")


@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return render_template('500.html'), 500


if __name__ == '__main__':
    # Set metadata
    APP_CREATION_DATE = "2025-04-02 15:08:59"
    APP_AUTHOR = "KishoreKumarKalli"

    # Start the application
    logger.info(f"Starting Brain AI Web Application v{APP_VERSION}")
    logger.info(f"Created: {APP_CREATION_DATE} by {APP_AUTHOR}")

    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)