#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brain MRI Analysis Platform - Main Application

This is the main entry point for the Brain MRI Analysis Platform web application.
It initializes and configures the Flask application, sets up necessary extensions,
and includes all routes for the web interface.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, render_template, flash, redirect, url_for, request, jsonify, send_from_directory
from flask_wtf.csrf import CSRFProtect
from flask_compress import Compress
from flask_caching import Cache
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix

# Import modules from the brain_segmentation_framework package
from brain_segmentation_framework.config import config
from brain_segmentation_framework.utils import file_utils, security_utils
from brain_segmentation_framework.web.forms import UploadForm, BatchProcessingForm, ContactForm
from brain_segmentation_framework.web.tasks import task_queue, process_mri_scan, get_task_status
from brain_segmentation_framework.processing import pipeline

# Load environment variables from .env file if it exists
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'logs', 'app.log'))
    ]
)
logger = logging.getLogger(__name__)


# Initialize Flask application
def create_app(config_name='default'):
    """
    Create and configure the Flask application.

    Args:
        config_name (str): The configuration to use. Defaults to 'default'.

    Returns:
        Flask: Configured Flask application instance.
    """
    app = Flask(__name__,
                static_folder=os.path.join(os.path.dirname(__file__), 'static'),
                template_folder=os.path.join(os.path.dirname(__file__), 'templates'))

    # Load configuration
    app_config = config.get_config(config_name)
    app.config.from_object(app_config)

    # Override config with environment variables if they exist
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', app.config.get('SECRET_KEY'))
    app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', app.config.get('UPLOAD_FOLDER'))
    app.config['RESULTS_FOLDER'] = os.environ.get('RESULTS_FOLDER', app.config.get('RESULTS_FOLDER'))

    # Ensure upload and results directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

    # Initialize extensions
    csrf = CSRFProtect(app)
    compress = Compress(app)
    cache = Cache(app)

    # Fix for running behind a proxy server
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

    # Register error handlers
    register_error_handlers(app)

    # Register blueprints if any
    # from brain_segmentation_framework.web.api import api_bp
    # app.register_blueprint(api_bp, url_prefix='/api')

    # Add context processors
    @app.context_processor
    def inject_globals():
        return {
            'current_year': datetime.now().year,
            'app_version': app.config.get('VERSION', '1.0.0'),
            'app_name': app.config.get('APP_NAME', 'Brain MRI Analysis Platform'),
            'author': 'KishoreKumarKalli'
        }

    # Register template filters
    @app.template_filter('format_date')
    def format_date(value, format='%Y-%m-%d %H:%M:%S'):
        if value is None:
            return ""
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value)
            except ValueError:
                return value
        return value.strftime(format)

    # After request handler
    @app.after_request
    def add_security_headers(response):
        """Add security headers to response"""
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        response.headers[
            'Content-Security-Policy'] = "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://unpkg.com https://cdn.plot.ly; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://unpkg.com; img-src 'self' data:; connect-src 'self'; font-src 'self' https://cdnjs.cloudflare.com; frame-src 'none'; object-src 'none'"
        return response

    # Register routes
    register_routes(app)

    logger.info(f"Application initialized with configuration: {config_name}")

    return app


def register_error_handlers(app):
    """
    Register error handlers for the Flask application.

    Args:
        app (Flask): The Flask application instance.
    """

    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('errors/404.html', error=e), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        logger.error(f"Internal server error: {e}", exc_info=True)
        return render_template('errors/500.html', error=e), 500

    @app.errorhandler(413)
    def request_entity_too_large(e):
        flash('The file is too large. Maximum file size is 1GB.', 'warning')
        return redirect(url_for('upload_file'))


def register_routes(app):
    """
    Register routes for the Flask application.

    Args:
        app (Flask): The Flask application instance.
    """

    @app.route('/')
    def index():
        """Home page route."""
        return render_template('index.html',
                               page_title="Home",
                               last_updated="2025-04-02 15:49:48")

    @app.route('/upload', methods=['GET', 'POST'])
    def upload_file():
        """Handle file upload route."""
        form = UploadForm()

        if form.validate_on_submit():
            try:
                # Check if a file was uploaded
                if 'file' not in request.files:
                    flash('No file part', 'warning')
                    return redirect(request.url)

                file = request.files['file']

                # If user does not select file, browser may submit an empty file
                if file.filename == '':
                    flash('No selected file', 'warning')
                    return redirect(request.url)

                # Check if the file is valid
                if file and file_utils.allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
                    # Secure the filename
                    filename = secure_filename(file.filename)

                    # Create a unique directory for this upload
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    upload_id = f"{timestamp}_{security_utils.generate_random_string(8)}"
                    upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], upload_id)
                    os.makedirs(upload_dir, exist_ok=True)

                    # Save the file
                    file_path = os.path.join(upload_dir, filename)
                    file.save(file_path)

                    # Save metadata (optional patient info)
                    metadata = {
                        'upload_id': upload_id,
                        'original_filename': filename,
                        'upload_timestamp': timestamp,
                        'file_size': os.path.getsize(file_path),
                        'patient_id': form.patientId.data,
                        'scan_date': form.scanDate.data.isoformat() if form.scanDate.data else None
                    }

                    # Save metadata to a JSON file
                    file_utils.save_json(os.path.join(upload_dir, 'metadata.json'), metadata)

                    # Process the file asynchronously
                    processing_options = {
                        'segmentation': True,
                        'abnormality_detection': form.abnormalityDetection.data,
                        'volumetric_analysis': form.volumetricAnalysis.data
                    }

                    # Enqueue the processing task
                    task_id = task_queue.enqueue(
                        process_mri_scan,
                        file_path=file_path,
                        upload_id=upload_id,
                        processing_options=processing_options,
                        metadata=metadata
                    )

                    # Redirect to processing status page
                    return redirect(url_for('processing_status', task_id=task_id))
                else:
                    allowed_ext = ', '.join(app.config['ALLOWED_EXTENSIONS'])
                    flash(f'Invalid file type. Allowed types: {allowed_ext}', 'warning')
                    return redirect(request.url)

            except Exception as e:
                logger.error(f"Error during file upload: {e}", exc_info=True)
                flash('An error occurred during file upload. Please try again.', 'danger')
                return redirect(request.url)

        return render_template('upload.html',
                               form=form,
                               page_title="Upload MRI Scan",
                               max_file_size=app.config.get('MAX_CONTENT_LENGTH', 1024 * 1024 * 1024),
                               last_updated="2025-04-02 15:49:48")

    @app.route('/batch-processing', methods=['POST'])
    def batch_processing():
        """Handle batch processing of multiple MRI files."""
        try:
            if 'files[]' not in request.files:
                return jsonify({'error': 'No files uploaded'}), 400

            files = request.files.getlist('files[]')

            if not files or files[0].filename == '':
                return jsonify({'error': 'No files selected'}), 400

            # Create a unique batch ID
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            batch_id = f"batch_{timestamp}_{security_utils.generate_random_string(8)}"
            batch_dir = os.path.join(app.config['UPLOAD_FOLDER'], batch_id)
            os.makedirs(batch_dir, exist_ok=True)

            # Save all files
            saved_files = []
            for file in files:
                if file and file_utils.allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(batch_dir, filename)
                    file.save(file_path)
                    saved_files.append({
                        'filename': filename,
                        'path': file_path,
                        'size': os.path.getsize(file_path)
                    })

            if not saved_files:
                return jsonify({'error': 'No valid files uploaded'}), 400

            # Get processing options from form
            process_types = request.form.getlist('process_types')
            processing_options = {
                'segmentation': 'segmentation' in process_types,
                'abnormality_detection': 'abnormality' in process_types,
                'volumetric_analysis': 'analysis' in process_types
            }

            batch_name = request.form.get('batch_name', f"Batch {timestamp}")

            # Save batch metadata
            metadata = {
                'batch_id': batch_id,
                'name': batch_name,
                'upload_timestamp': timestamp,
                'file_count': len(saved_files),
                'files': saved_files,
                'processing_options': processing_options
            }

            file_utils.save_json(os.path.join(batch_dir, 'batch_metadata.json'), metadata)

            # Enqueue batch processing task
            task_id = task_queue.enqueue(
                pipeline.process_batch,
                batch_dir=batch_dir,
                batch_id=batch_id,
                files=saved_files,
                processing_options=processing_options,
                metadata=metadata
            )

            return jsonify({
                'success': True,
                'batch_id': batch_id,
                'task_id': task_id,
                'file_count': len(saved_files),
                'redirect_url': url_for('batch_status', task_id=task_id)
            }), 200

        except Exception as e:
            logger.error(f"Error during batch processing: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500

    @app.route('/processing-status/<task_id>')
    def processing_status(task_id):
        """Display the processing status page for a single file."""
        task_status = get_task_status(task_id)

        if not task_status:
            flash('Invalid or expired task ID', 'warning')
            return redirect(url_for('upload_file'))

        return render_template('processing_status.html',
                               task_id=task_id,
                               task_status=task_status,
                               page_title="Processing Status",
                               refresh_interval=app.config.get('STATUS_REFRESH_INTERVAL', 5000),
                               last_updated="2025-04-02 15:55:40")

    @app.route('/batch-status/<task_id>')
    def batch_status(task_id):
        """Display the processing status page for a batch of files."""
        task_status = get_task_status(task_id)

        if not task_status:
            flash('Invalid or expired batch task ID', 'warning')
            return redirect(url_for('upload_file'))

        return render_template('batch_status.html',
                               task_id=task_id,
                               task_status=task_status,
                               page_title="Batch Processing Status",
                               refresh_interval=app.config.get('STATUS_REFRESH_INTERVAL', 5000),
                               last_updated="2025-04-02 15:55:40")

    @app.route('/api/task-status/<task_id>')
    def api_task_status(task_id):
        """API endpoint to get the current status of a task."""
        task_status = get_task_status(task_id)

        if not task_status:
            return jsonify({'error': 'Task not found'}), 404

        return jsonify(task_status)

    @app.route('/results/<task_id>')
    def view_results(task_id):
        """Display analysis results for a completed task."""
        task_status = get_task_status(task_id)

        if not task_status or task_status.get('status') != 'completed':
            flash('Results not available or processing not completed', 'warning')
            return redirect(url_for('upload_file'))

        # Get the results data
        results_path = task_status.get('results_path')
        if not results_path or not os.path.exists(results_path):
            flash('Results data not found', 'warning')
            return redirect(url_for('upload_file'))

        # Load results data
        results_data = file_utils.load_json(os.path.join(results_path, 'results.json'))

        return render_template('results.html',
                               task_id=task_id,
                               results=results_data,
                               page_title="Analysis Results",
                               last_updated="2025-04-02 15:55:40")

    @app.route('/batch-results/<batch_id>')
    def view_batch_results(batch_id):
        """Display analysis results for a completed batch processing task."""
        # Load batch results summary
        batch_dir = os.path.join(app.config['RESULTS_FOLDER'], batch_id)

        if not os.path.exists(batch_dir):
            flash('Batch results not found', 'warning')
            return redirect(url_for('upload_file'))

        # Load batch summary
        summary_path = os.path.join(batch_dir, 'batch_summary.json')
        if not os.path.exists(summary_path):
            flash('Batch summary not found', 'warning')
            return redirect(url_for('upload_file'))

        batch_summary = file_utils.load_json(summary_path)

        return render_template('batch_results.html',
                               batch_id=batch_id,
                               summary=batch_summary,
                               page_title="Batch Analysis Results",
                               last_updated="2025-04-02 15:55:40")

    @app.route('/download/<path:filename>')
    def download_file(filename):
        """Handle file downloads."""
        # Validate the requested path to prevent directory traversal
        requested_path = os.path.abspath(os.path.join(app.config['RESULTS_FOLDER'], filename))
        if not requested_path.startswith(os.path.abspath(app.config['RESULTS_FOLDER'])):
            flash('Invalid download request', 'danger')
            return redirect(url_for('index'))

        directory = os.path.dirname(requested_path)
        file = os.path.basename(requested_path)

        return send_from_directory(directory, file, as_attachment=True)

    @app.route('/documentation')
    def documentation():
        """Display documentation page."""
        return render_template('documentation.html',
                               page_title="Documentation",
                               last_updated="2025-04-02 15:55:40")

    @app.route('/api-docs')
    def api_docs():
        """Display API documentation page."""
        return render_template('api_docs.html',
                               page_title="API Documentation",
                               last_updated="2025-04-02 15:55:40")

    @app.route('/about')
    def about():
        """Display about page."""
        return render_template('about.html',
                               page_title="About",
                               last_updated="2025-04-02 15:55:40")

    @app.route('/contact', methods=['GET', 'POST'])
    def contact():
        """Handle contact form."""
        form = ContactForm()

        if form.validate_on_submit():
            # Process the form data
            contact_data = {
                'name': form.name.data,
                'email': form.email.data,
                'subject': form.subject.data,
                'message': form.message.data,
                'timestamp': datetime.now().isoformat()
            }

            try:
                # Save contact submission
                contacts_dir = os.path.join(os.path.dirname(__file__), 'data', 'contacts')
                os.makedirs(contacts_dir, exist_ok=True)

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                contact_id = f"{timestamp}_{security_utils.generate_random_string(8)}"
                contact_file = os.path.join(contacts_dir, f"{contact_id}.json")

                file_utils.save_json(contact_file, contact_data)

                # Send email notification (would need to implement this)
                # send_contact_notification(contact_data)

                flash('Thank you for your message! We will get back to you soon.', 'success')
                return redirect(url_for('index'))

            except Exception as e:
                logger.error(f"Error processing contact form: {e}", exc_info=True)
                flash('An error occurred while processing your message. Please try again.', 'danger')

        return render_template('contact.html',
                               form=form,
                               page_title="Contact Us",
                               last_updated="2025-04-02 15:55:40")

    @app.route('/user-guide')
    def user_guide():
        """Display user guide page."""
        return render_template('user_guide.html',
                               page_title="User Guide",
                               last_updated="2025-04-02 15:55:40")

    @app.route('/privacy-policy')
    def privacy_policy():
        """Display privacy policy page."""
        return render_template('privacy_policy.html',
                               page_title="Privacy Policy",
                               last_updated="2025-04-02 15:55:40")

    @app.route('/terms-of-service')
    def terms_of_service():
        """Display terms of service page."""
        return render_template('terms_of_service.html',
                               page_title="Terms of Service",
                               last_updated="2025-04-02 15:55:40")

    @app.route('/faq')
    def faq():
        """Display FAQ page."""
        return render_template('faq.html',
                               page_title="Frequently Asked Questions",
                               last_updated="2025-04-02 15:55:40")


# Create the Flask application using the factory function
app = create_app(os.environ.get('FLASK_CONFIG', 'default'))

if __name__ == '__main__':
    # Only for development - use a proper WSGI server in production
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'production') == 'development'

    app.run(host='0.0.0.0', port=port, debug=debug)