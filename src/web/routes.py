"""
Routes module for Brain MRI Analysis Platform.
This module defines all the Flask routes and view functions for the web application.
"""

import os
import sys
import logging
import nibabel as nib
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Blueprint, render_template, request, redirect, url_for, flash, send_file, jsonify, session
from werkzeug.utils import secure_filename
import json
import uuid
import io
import zipfile
from pathlib import Path
import shutil
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('web_routes')

# Create Blueprint
routes_bp = Blueprint('routes', __name__)

# Metadata
ROUTES_VERSION = "1.0.0"

# Import task processing functions
from .tasks import (
    task_queue,
    tasks,
    load_nifti_file,
    allowed_file
)


@routes_bp.route('/')
def index():
    """Render the home page."""
    return render_template('index.html',
                           version=ROUTES_VERSION,
                           creation_date=ROUTES_CREATION_DATE,
                           author=ROUTES_AUTHOR)


@routes_bp.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html',
                           version=ROUTES_VERSION,
                           creation_date=ROUTES_CREATION_DATE,
                           author=ROUTES_AUTHOR)


@routes_bp.route('/upload', methods=['GET', 'POST'])
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

        # Get upload folder from app config
        upload_folder = current_app.config['UPLOAD_FOLDER']
        allowed_extensions = current_app.config['ALLOWED_EXTENSIONS']

        # Check if file type is allowed
        if file and allowed_file(file.filename, allowed_extensions):
            # Generate unique ID for this session/task
            session_id = str(uuid.uuid4())

            # Save the file
            filename = secure_filename(file.filename)
            file_path = os.path.join(upload_folder, f"{session_id}_{filename}")
            file.save(file_path)

            # Store file path in session
            session['file_path'] = file_path
            session['filename'] = filename
            session['session_id'] = session_id

            # Redirect to processing options
            return redirect(url_for('routes.processing_options'))
        else:
            flash(f'Invalid file type. Allowed types: {", ".join(allowed_extensions)}')
            return redirect(request.url)

    # GET request - render upload form
    return render_template('upload.html')


@routes_bp.route('/processing-options', methods=['GET'])
def processing_options():
    """Show processing options for the uploaded file."""
    if 'file_path' not in session:
        flash('No file uploaded. Please upload a file first.')
        return redirect(url_for('routes.upload_file'))

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
            upload_folder = current_app.config['UPLOAD_FOLDER']
            preview_path = os.path.join(upload_folder, f"preview_{session.get('session_id')}.png")
            Image.fromarray(middle_slice).save(preview_path)

            # Make path relative to static
            preview_path = os.path.relpath(preview_path, os.path.join(current_app.root_path, 'static'))
        except Exception as e:
            logger.error(f"Error generating preview: {str(e)}")
            preview_path = None

    return render_template('processing_options.html',
                           filename=filename,
                           preview_path=preview_path)


@routes_bp.route('/run-segmentation', methods=['POST'])
def run_segmentation_task():
    """Run segmentation on the uploaded file."""
    if 'file_path' not in session:
        return jsonify({'error': 'No file uploaded'}), 400

    file_path = session.get('file_path')
    session_id = session.get('session_id')

    # Create output directory
    results_folder = current_app.config['RESULTS_FOLDER']
    output_dir = os.path.join(results_folder, session_id, 'segmentation')
    os.makedirs(output_dir, exist_ok=True)

    # Create task
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        'type': 'segmentation',
        'status': 'queued',
        'progress': 0,
        'session_id': session_id,
        'created_at': ROUTES_CREATION_DATE,
        'created_by': ROUTES_AUTHOR
    }

    # Add task to queue
    task_queue.put((task_id, 'segmentation', (file_path, output_dir)))

    return jsonify({
        'task_id': task_id,
        'status': 'queued',
        'message': 'Segmentation task queued'
    })


@routes_bp.route('/run-abnormality', methods=['POST'])
def run_abnormality_task():
    """Run abnormality detection on the uploaded file."""
    if 'file_path' not in session:
        return jsonify({'error': 'No file uploaded'}), 400

    file_path = session.get('file_path')
    session_id = session.get('session_id')

    # Create output directory
    results_folder = current_app.config['RESULTS_FOLDER']
    output_dir = os.path.join(results_folder, session_id, 'abnormality')
    os.makedirs(output_dir, exist_ok=True)

    # Create task
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        'type': 'abnormality',
        'status': 'queued',
        'progress': 0,
        'session_id': session_id,
        'created_at': ROUTES_CREATION_DATE,
        'created_by': ROUTES_AUTHOR
    }

    # Add task to queue
    task_queue.put((task_id, 'abnormality', (file_path, output_dir)))

    return jsonify({
        'task_id': task_id,
        'status': 'queued',
        'message': 'Abnormality detection task queued'
    })


@routes_bp.route('/run-analysis', methods=['POST'])
def run_analysis_task():
    """Run volumetric analysis on segmentation results."""
    if 'file_path' not in session:
        return jsonify({'error': 'No file uploaded'}), 400

    session_id = session.get('session_id')
    file_path = session.get('file_path')

    # Check if segmentation results exist
    results_folder = current_app.config['RESULTS_FOLDER']
    seg_path = os.path.join(results_folder, session_id, 'segmentation', 'segmentation_result.nii.gz')
    if not os.path.exists(seg_path):
        return jsonify({
            'error': 'Segmentation results not found. Run segmentation first.'
        }), 400

    # Create output directory
    output_dir = os.path.join(results_folder, session_id, 'analysis')
    os.makedirs(output_dir, exist_ok=True)

    # Create task
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        'type': 'analysis',
        'status': 'queued',
        'progress': 0,
        'session_id': session_id,
        'created_at': "2025-04-02 15:11:41",
        'created_by': "KishoreKumarKalli"
    }

    # Add task to queue
    task_queue.put((task_id, 'analysis', (seg_path, file_path, output_dir)))

    return jsonify({
        'task_id': task_id,
        'status': 'queued',
        'message': 'Analysis task queued'
    })


@routes_bp.route('/run-report', methods=['POST'])
def run_report_task():
    """Generate a comprehensive report from all results."""
    if 'session_id' not in session:
        return jsonify({'error': 'No active session'}), 400

    session_id = session.get('session_id')
    results_folder = current_app.config['RESULTS_FOLDER']

    # Check if necessary results exist
    seg_dir = os.path.join(results_folder, session_id, 'segmentation')
    analysis_dir = os.path.join(results_folder, session_id, 'analysis')

    if not os.path.exists(seg_dir) or not os.path.exists(analysis_dir):
        return jsonify({
            'error': 'Missing required results. Run segmentation and analysis first.'
        }), 400

    # Create output directory for report
    output_dir = os.path.join(results_folder, session_id, 'report')
    os.makedirs(output_dir, exist_ok=True)

    # Create task
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        'type': 'report',
        'status': 'queued',
        'progress': 0,
        'session_id': session_id,
        'created_at': "2025-04-02 15:11:41",
        'created_by': "KishoreKumarKalli"
    }

    # Add task to queue
    task_queue.put((task_id, 'report', (session_id, output_dir)))

    return jsonify({
        'task_id': task_id,
        'status': 'queued',
        'message': 'Report generation task queued'
    })


@routes_bp.route('/task-status/<task_id>', methods=['GET'])
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


@routes_bp.route('/results/<session_id>', methods=['GET'])
def show_results(session_id):
    """Show results for a session."""
    # Check if results exist
    results_folder = current_app.config['RESULTS_FOLDER']
    session_dir = os.path.join(results_folder, session_id)
    if not os.path.exists(session_dir):
        flash('No results found for this session')
        return redirect(url_for('routes.index'))

    # Collect all results
    results = {
        'session_id': session_id,
        'metadata': {
            'date': "2025-04-02 15:11:41",
            'author': "KishoreKumarKalli"
        },
        'segmentation': None,
        'abnormality': None,
        'analysis': None,
        'report': None
    }

    # Check for segmentation results
    seg_dir = os.path.join(session_dir, 'segmentation')
    if os.path.exists(seg_dir):
        viz_file = os.path.join(seg_dir, 'segmentation_visualization.png')
        if os.path.exists(viz_file):
            results['segmentation'] = {
                'visualization': os.path.relpath(viz_file, os.path.join(current_app.root_path, 'static')),
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
                'visualization': os.path.relpath(viz_file, os.path.join(current_app.root_path, 'static')),
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
                'chart': os.path.relpath(chart_file, os.path.join(current_app.root_path, 'static')),
                'download_path': f"/download/{session_id}/analysis",
                'volumes_table': volumes_table
            }

    # Check for report
    report_dir = os.path.join(session_dir, 'report')
    if os.path.exists(report_dir):
        report_file = os.path.join(report_dir, 'brain_analysis_report.pdf')
        if os.path.exists(report_file):
            results['report'] = {
                'download_path': f"/download-report/{session_id}"
            }

    return render_template('results.html', results=results)


@routes_bp.route('/download/<session_id>/<result_type>', methods=['GET'])
def download_results(session_id, result_type):
    """Download results as a zip file."""
    # Check if results exist
    results_folder = current_app.config['RESULTS_FOLDER']
    result_dir = os.path.join(results_folder, session_id, result_type)
    if not os.path.exists(result_dir):
        flash(f'No {result_type} results found for this session')
        return redirect(url_for('routes.index'))

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


@routes_bp.route('/download-report/<session_id>', methods=['GET'])
def download_report(session_id):
    """Download the analysis report as a PDF."""
    # Check if report exists
    results_folder = current_app.config['RESULTS_FOLDER']
    report_file = os.path.join(results_folder, session_id, 'report', 'brain_analysis_report.pdf')

    if not os.path.exists(report_file):
        flash('Report not found for this session')
        return redirect(url_for('routes.index'))

    return send_file(
        report_file,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'brain_ai_report_{session_id}.pdf'
    )


@routes_bp.route('/documentation')
def documentation():
    """Show documentation page."""
    return render_template('documentation.html',
                           version=ROUTES_VERSION,
                           creation_date="2025-04-02 15:11:41",
                           author="KishoreKumarKalli")


@routes_bp.route('/api/documentation')
def api_documentation():
    """Show API documentation page."""
    return render_template('api_documentation.html',
                           version=ROUTES_VERSION,
                           creation_date="2025-04-02 15:11:41",
                           author="KishoreKumarKalli")


@routes_bp.route('/batch-processing', methods=['GET', 'POST'])
def batch_processing():
    """Handle batch processing of multiple files."""
    if request.method == 'POST':
        # Check if files were uploaded
        if 'files[]' not in request.files:
            flash('No files part')
            return redirect(request.url)

        files = request.files.getlist('files[]')

        # Check if any files were selected
        if len(files) == 0 or files[0].filename == '':
            flash('No files selected')
            return redirect(request.url)

        # Generate batch ID
        batch_id = str(uuid.uuid4())

        # Get allowed extensions
        allowed_extensions = current_app.config['ALLOWED_EXTENSIONS']

        # Save each valid file
        saved_files = []
        upload_folder = current_app.config['UPLOAD_FOLDER']
        batch_folder = os.path.join(upload_folder, f"batch_{batch_id}")
        os.makedirs(batch_folder, exist_ok=True)

        for file in files:
            if file and allowed_file(file.filename, allowed_extensions):
                filename = secure_filename(file.filename)
                file_path = os.path.join(batch_folder, filename)
                file.save(file_path)
                saved_files.append({
                    'filename': filename,
                    'path': file_path
                })

        if not saved_files:
            flash(f'No valid files were uploaded. Allowed types: {", ".join(allowed_extensions)}')
            return redirect(request.url)

        # Store batch info in session
        session['batch_id'] = batch_id
        session['batch_files'] = saved_files

        # Create a batch task
        task_id = str(uuid.uuid4())

        # Get processing options from form
        process_types = request.form.getlist('process_types')

        if not process_types:
            flash('No processing types selected')
            return redirect(request.url)

        # Create task
        tasks[task_id] = {
            'type': 'batch',
            'status': 'queued',
            'progress': 0,
            'batch_id': batch_id,
            'process_types': process_types,
            'created_at': "2025-04-02 15:11:41",
            'created_by': "KishoreKumarKalli",
            'files': saved_files
        }

        # Add batch task to queue
        results_folder = current_app.config['RESULTS_FOLDER']
        batch_results_dir = os.path.join(results_folder, f"batch_{batch_id}")
        task_queue.put((task_id, 'batch', (saved_files, process_types, batch_results_dir)))

        # Redirect to batch status page
        return redirect(url_for('routes.batch_status', task_id=task_id))

    # GET request - render batch upload form
    return render_template('batch_processing.html')


@routes_bp.route('/batch-status/<task_id>', methods=['GET'])
def batch_status(task_id):
    """Show status of a batch processing task."""
    if task_id not in tasks:
        flash('Batch task not found')
        return redirect(url_for('routes.batch_processing'))

    task = tasks[task_id]
    return render_template('batch_status.html', task=task)


@routes_bp.route('/batch-results/<batch_id>', methods=['GET'])
def batch_results(batch_id):
    """Show results for a batch processing job."""
    results_folder = current_app.config['RESULTS_FOLDER']
    batch_dir = os.path.join(results_folder, f"batch_{batch_id}")

    if not os.path.exists(batch_dir):
        flash('No results found for this batch')
        return redirect(url_for('routes.index'))

    # Collect all batch results
    batch_results = {
        'batch_id': batch_id,
        'metadata': {
            'date': "2025-04-02 15:11:41",
            'author': "KishoreKumarKalli"
        },
        'files': []
    }

    # Look for result summary file
    summary_path = os.path.join(batch_dir, 'batch_summary.json')
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                summary_data = json.load(f)
                batch_results['summary'] = summary_data
        except Exception as e:
            logger.error(f"Error loading batch summary: {str(e)}")

    # Get individual file results
    file_dirs = [d for d in os.listdir(batch_dir) if os.path.isdir(os.path.join(batch_dir, d))]

    for file_dir in file_dirs:
        file_result = {
            'file_id': file_dir,
            'filename': file_dir,  # This might be refined from a metadata file
            'results': {}
        }

        # Check for different result types
        for result_type in ['segmentation', 'abnormality', 'analysis']:
            result_path = os.path.join(batch_dir, file_dir, result_type)
            if os.path.exists(result_path):
                file_result['results'][result_type] = {
                    'available': True,
                    'download_path': f"/download-batch/{batch_id}/{file_dir}/{result_type}"
                }

        batch_results['files'].append(file_result)

    return render_template('batch_results.html', results=batch_results)


@routes_bp.route('/download-batch/<batch_id>/<file_id>/<result_type>', methods=['GET'])
def download_batch_results(batch_id, file_id, result_type):
    """Download results for a specific file in a batch."""
    results_folder = current_app.config['RESULTS_FOLDER']
    result_dir = os.path.join(results_folder, f"batch_{batch_id}", file_id, result_type)

    if not os.path.exists(result_dir):
        flash(f'No {result_type} results found for this file')
        return redirect(url_for('routes.batch_results', batch_id=batch_id))

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
        download_name=f'brain_ai_batch_{batch_id}_{file_id}_{result_type}.zip'
    )


@routes_bp.route('/download-complete-batch/<batch_id>', methods=['GET'])
def download_complete_batch(batch_id):
    """Download all results for a batch job."""
    results_folder = current_app.config['RESULTS_FOLDER']
    batch_dir = os.path.join(results_folder, f"batch_{batch_id}")

    if not os.path.exists(batch_dir):
        flash('No results found for this batch')
        return redirect(url_for('routes.index'))

    # Create a zip file in memory
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        for root, dirs, files in os.walk(batch_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, batch_dir)
                zf.write(file_path, arcname)

    # Reset file pointer
    memory_file.seek(0)

    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'brain_ai_complete_batch_{batch_id}.zip'
    )


# Error handlers
@routes_bp.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('404.html',
                           version=ROUTES_VERSION,
                           creation_date="2025-04-02 15:11:41",
                           author="KishoreKumarKalli"), 404


@routes_bp.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return render_template('500.html',
                           version=ROUTES_VERSION,
                           author="KishoreKumarKalli"), 500