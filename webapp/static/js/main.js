/**
 * Brain AI - Neuroimaging Analysis Web Application
 * Main JavaScript Module
 *
 * This file handles:
 * - 3D visualization of MRI data and segmentation results
 * - AJAX requests to Flask backend endpoints
 * - UI interactions and form handling
 * - Results visualization and statistical displays
 */

// Global variables
let viewer = null;
let currentSubjectData = null;
let segmentationOverlay = null;
let analysisResults = null;

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeUI();
    setupEventListeners();

    // Check if we have a subject ID in the URL (for direct linking)
    const urlParams = new URLSearchParams(window.location.search);
    const subjectId = urlParams.get('subject_id');
    if (subjectId) {
        loadSubject(subjectId);
    }
});

/**
 * Initialize UI components and visualization containers
 */
function initializeUI() {
    // Initialize the brain viewer (using Papaya.js or other neuroimaging library)
    initializeViewer();

    // Initialize chart containers for statistics (using Chart.js)
    initializeCharts();

    // Initialize form components
    resetForms();

    // Show loading spinner initially
    showLoadingSpinner(false);
}

/**
 * Initialize the 3D brain viewer component
 */
function initializeViewer() {
    const viewerContainer = document.getElementById('brain-viewer');
    if (!viewerContainer) return;

    // Options for the viewer - adjust based on your requirements
    const viewerOptions = {
        worldSpace: true,
        expandable: true,
        smoothDisplay: true,
        coordinateSpace: "world"
    };

    // Initialize papaya (or another 3D brain visualization library)
    // This is a placeholder - replace with actual initialization code based on your chosen library
    if (typeof papaya !== 'undefined') {
        papaya.Container.startPapaya();
        papaya.Container.addViewer('brain-viewer', viewerOptions);
        viewer = papaya.Container.viewers[0];
    } else {
        console.error('Visualization library not loaded');
        viewerContainer.innerHTML = '<div class="error-message">Visualization library failed to load</div>';
    }
}

/**
 * Initialize charts for statistical visualization
 */
function initializeCharts() {
    // Volume comparison chart
    const volumeCtx = document.getElementById('volume-chart');
    if (volumeCtx) {
        new Chart(volumeCtx, {
            type: 'bar',
            data: {
                labels: ['Gray Matter', 'White Matter', 'CSF', 'Hippocampus'],
                datasets: [{
                    label: 'Volume (cc)',
                    data: [0, 0, 0, 0], // Placeholder data
                    backgroundColor: [
                        'rgba(54, 162, 235, 0.5)',
                        'rgba(255, 206, 86, 0.5)',
                        'rgba(75, 192, 192, 0.5)',
                        'rgba(153, 102, 255, 0.5)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Volume (cubic cm)'
                        }
                    }
                }
            }
        });
    }

    // Abnormality probability chart
    const abnormalityCtx = document.getElementById('abnormality-chart');
    if (abnormalityCtx) {
        new Chart(abnormalityCtx, {
            type: 'radar',
            data: {
                labels: ['Atrophy', 'Lesions', 'Asymmetry', 'Ventricle Enlargement'],
                datasets: [{
                    label: 'Probability',
                    data: [0, 0, 0, 0], // Placeholder data
                    fill: true,
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderColor: 'rgb(255, 99, 132)',
                    pointBackgroundColor: 'rgb(255, 99, 132)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgb(255, 99, 132)'
                }]
            },
            options: {
                elements: {
                    line: {
                        borderWidth: 3
                    }
                },
                scales: {
                    r: {
                        angleLines: {
                            display: true
                        },
                        suggestedMin: 0,
                        suggestedMax: 1
                    }
                }
            }
        });
    }
}

/**
 * Set up all event listeners for UI interactions
 */
function setupEventListeners() {
    // Subject selection form
    const subjectForm = document.getElementById('subject-form');
    if (subjectForm) {
        subjectForm.addEventListener('submit', function(event) {
            event.preventDefault();
            const subjectId = document.getElementById('subject-id').value;
            if (subjectId) {
                loadSubject(subjectId);
            }
        });
    }

    // File upload form
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(event) {
            event.preventDefault();
            uploadMRI();
        });
    }

    // Segmentation button
    const segmentBtn = document.getElementById('segment-btn');
    if (segmentBtn) {
        segmentBtn.addEventListener('click', function() {
            runSegmentation();
        });
    }

    // Analysis button
    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', function() {
            runAnalysis();
        });
    }

    // Toggle overlay buttons
    const toggleBtns = document.querySelectorAll('.toggle-overlay');
    toggleBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            toggleOverlay(this.dataset.overlay);
        });
    });

    // View mode buttons
    const viewModeBtns = document.querySelectorAll('.view-mode');
    viewModeBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            setViewMode(this.dataset.mode);
        });
    });

    // Report generation button
    const reportBtn = document.getElementById('generate-report');
    if (reportBtn) {
        reportBtn.addEventListener('click', function() {
            generateReport();
        });
    }
}

/**
 * Load a subject's MRI data by ID
 * @param {string} subjectId - ID of the subject to load
 */
function loadSubject(subjectId) {
    showLoadingSpinner(true);

    // Make API request to get subject data
    fetch(`/api/subject/${subjectId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Subject not found');
            }
            return response.json();
        })
        .then(data => {
            currentSubjectData = data;
            updateSubjectInfo(data);
            loadMRIData(data.mri_path);

            // If segmentation already exists, load it
            if (data.has_segmentation) {
                loadSegmentation(data.segmentation_path);
            }

            // If analysis results exist, load them
            if (data.has_analysis) {
                loadAnalysisResults(data.analysis_path);
            }

            showLoadingSpinner(false);
        })
        .catch(error => {
            console.error('Error loading subject:', error);
            showError(`Failed to load subject: ${error.message}`);
            showLoadingSpinner(false);
        });
}

/**
 * Upload a new MRI scan
 */
function uploadMRI() {
    const fileInput = document.getElementById('mri-file');
    if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
        showError('Please select a file to upload');
        return;
    }

    const formData = new FormData();
    formData.append('mri_file', fileInput.files[0]);

    // Add metadata if available
    const subjectInfo = document.getElementById('subject-info');
    if (subjectInfo) {
        formData.append('subject_info', subjectInfo.value);
    }

    const diagnosisSelect = document.getElementById('diagnosis');
    if (diagnosisSelect) {
        formData.append('diagnosis', diagnosisSelect.value);
    }

    showLoadingSpinner(true);

    // Send file to server
    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showMessage('File uploaded successfully');
            loadSubject(data.subject_id);
        } else {
            showError(data.error || 'Upload failed');
        }
        showLoadingSpinner(false);
    })
    .catch(error => {
        console.error('Error uploading file:', error);
        showError('Upload failed. Please try again.');
        showLoadingSpinner(false);
    });
}

/**
 * Run segmentation on the current MRI
 */
function runSegmentation() {
    if (!currentSubjectData) {
        showError('Please load a subject first');
        return;
    }

    showLoadingSpinner(true);
    showMessage('Running brain segmentation...');

    // Call the segmentation API
    fetch(`/api/segment/${currentSubjectData.id}`, {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showMessage('Segmentation completed');
            loadSegmentation(data.segmentation_path);
        } else {
            showError(data.error || 'Segmentation failed');
        }
        showLoadingSpinner(false);
    })
    .catch(error => {
        console.error('Error during segmentation:', error);
        showError('Segmentation failed. Please try again.');
        showLoadingSpinner(false);
    });
}

/**
 * Run analysis on the segmented MRI
 */
function runAnalysis() {
    if (!currentSubjectData || !currentSubjectData.has_segmentation) {
        showError('Please run segmentation first');
        return;
    }

    showLoadingSpinner(true);
    showMessage('Running analysis...');

    // Call the analysis API
    fetch(`/api/analyze/${currentSubjectData.id}`, {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showMessage('Analysis completed');
            loadAnalysisResults(data.analysis_path);
        } else {
            showError(data.error || 'Analysis failed');
        }
        showLoadingSpinner(false);
    })
    .catch(error => {
        console.error('Error during analysis:', error);
        showError('Analysis failed. Please try again.');
        showLoadingSpinner(false);
    });
}

/**
 * Load MRI data into the viewer
 * @param {string} mriPath - Path to the MRI file
 */
function loadMRIData(mriPath) {
    if (!viewer) return;

    // Clear existing data
    if (typeof papaya !== 'undefined') {
        // This is a placeholder - replace with actual loading code based on your chosen library
        papaya.Container.addImage(0, mriPath);
    } else {
        console.error('Visualization library not loaded');
    }
}

/**
 * Load segmentation data as an overlay
 * @param {string} segmentationPath - Path to the segmentation file
 */
function loadSegmentation(segmentationPath) {
    if (!viewer) return;

    // Add segmentation as overlay
    if (typeof papaya !== 'undefined') {
        // This is a placeholder - replace with actual overlay code based on your chosen library
        papaya.Container.addImage(0, segmentationPath, {
            lut: 'Overlay',
            alpha: 0.5,
            colorTable: 'Spectrum'
        });
        segmentationOverlay = true;

        // Update UI to reflect segmentation is loaded
        const segmentBtn = document.getElementById('segment-btn');
        if (segmentBtn) {
            segmentBtn.textContent = 'Segmentation Complete';
            segmentBtn.disabled = true;
        }

        const analyzeBtn = document.getElementById('analyze-btn');
        if (analyzeBtn) {
            analyzeBtn.disabled = false;
        }
    }
}

/**
 * Load analysis results and update charts
 * @param {string} analysisPath - Path to the analysis results
 */
function loadAnalysisResults(analysisPath) {
    fetch(analysisPath)
        .then(response => response.json())
        .then(data => {
            analysisResults = data;

            // Update volume chart
            updateVolumeChart(data.volumes);

            // Update abnormality chart
            updateAbnormalityChart(data.abnormalities);

            // Show statistical results
            displayStatistics(data.statistics);

            // Update UI to reflect analysis is loaded
            const analyzeBtn = document.getElementById('analyze-btn');
            if (analyzeBtn) {
                analyzeBtn.textContent = 'Analysis Complete';
                analyzeBtn.disabled = true;
            }

            const reportBtn = document.getElementById('generate-report');
            if (reportBtn) {
                reportBtn.disabled = false;
            }
        })
        .catch(error => {
            console.error('Error loading analysis results:', error);
            showError('Failed to load analysis results.');
        });
}

/**
 * Update the volume chart with segmentation volumes
 * @param {Object} volumes - Volume data for brain regions
 */
function updateVolumeChart(volumes) {
    const volumeChart = Chart.getChart('volume-chart');
    if (!volumeChart) return;

    volumeChart.data.labels = Object.keys(volumes);
    volumeChart.data.datasets[0].data = Object.values(volumes);
    volumeChart.update();
}

/**
 * Update the abnormality chart with detection results
 * @param {Object} abnormalities - Abnormality detection data
 */
function updateAbnormalityChart(abnormalities) {
    const abnormalityChart = Chart.getChart('abnormality-chart');
    if (!abnormalityChart) return;

    abnormalityChart.data.labels = Object.keys(abnormalities);
    abnormalityChart.data.datasets[0].data = Object.values(abnormalities);
    abnormalityChart.update();
}

/**
 * Display statistical analysis results
 * @param {Object} statistics - Statistical analysis data
 */
function displayStatistics(statistics) {
    const statsContainer = document.getElementById('statistics-container');
    if (!statsContainer) return;

    let html = '<h3>Statistical Analysis</h3>';

    // Display p-values and other statistics
    if (statistics.p_values) {
        html += '<div class="stat-section"><h4>Significance Testing</h4><table class="stats-table">';
        html += '<tr><th>Comparison</th><th>p-value</th><th>Significance</th></tr>';

        for (const [test, value] of Object.entries(statistics.p_values)) {
            const significant = value < 0.05 ? 'Significant' : 'Not significant';
            const rowClass = value < 0.05 ? 'significant' : '';
            html += `<tr class="${rowClass}"><td>${test}</td><td>${value.toFixed(4)}</td><td>${significant}</td></tr>`;
        }

        html += '</table></div>';
    }

    // Display correlations
    if (statistics.correlations) {
        html += '<div class="stat-section"><h4>Clinical Correlations</h4><table class="stats-table">';
        html += '<tr><th>Measure</th><th>Correlation (r)</th><th>Strength</th></tr>';

        for (const [measure, correlation] of Object.entries(statistics.correlations)) {
            const strength = getCorrelationStrength(correlation);
            const rowClass = Math.abs(correlation) > 0.5 ? 'strong-correlation' : '';
            html += `<tr class="${rowClass}"><td>${measure}</td><td>${correlation.toFixed(2)}</td><td>${strength}</td></tr>`;
        }

        html += '</table></div>';
    }

    // Display classification results
    if (statistics.classification) {
        html += '<div class="stat-section"><h4>Classification Results</h4>';
        html += `<p><strong>Predicted Class:</strong> ${statistics.classification.predicted_class}</p>`;
        html += `<p><strong>Confidence:</strong> ${(statistics.classification.confidence * 100).toFixed(1)}%</p>`;
        html += '</div>';
    }

    statsContainer.innerHTML = html;
}

/**
 * Get text description of correlation strength
 * @param {number} correlation - Correlation coefficient
 * @returns {string} Description of correlation strength
 */
function getCorrelationStrength(correlation) {
    const abs = Math.abs(correlation);
    if (abs >= 0.8) return 'Very Strong';
    if (abs >= 0.6) return 'Strong';
    if (abs >= 0.4) return 'Moderate';
    if (abs >= 0.2) return 'Weak';
    return 'Very Weak';
}

/**
 * Toggle visibility of segmentation overlay
 * @param {string} overlayName - Name of the overlay to toggle
 */
function toggleOverlay(overlayName) {
    if (!viewer || !segmentationOverlay) return;

    // This is a placeholder - replace with actual toggle code based on your chosen library
    if (typeof papaya !== 'undefined') {
        const currentVisibility = papaya.Container.getOverlayVisibility(0, 1);
        papaya.Container.setOverlayVisibility(0, 1, !currentVisibility);
    }
}

/**
 * Change the view mode (axial, coronal, sagittal, or 3D)
 * @param {string} mode - View mode to set
 */
function setViewMode(mode) {
    if (!viewer) return;

    // This is a placeholder - replace with actual view mode code based on your chosen library
    if (typeof papaya !== 'undefined') {
        switch(mode) {
            case 'axial':
                papaya.Container.viewers[0].gotoCoordinate(0, 0, 0, 0);
                break;
            case 'coronal':
                papaya.Container.viewers[0].gotoCoordinate(0, 0, 0, 1);
                break;
            case 'sagittal':
                papaya.Container.viewers[0].gotoCoordinate(0, 0, 0, 2);
                break;
            case '3d':
                // If your library supports 3D rendering
                console.log('3D mode not implemented');
                break;
        }
    }
}

/**
 * Generate a downloadable report
 */
function generateReport() {
    if (!currentSubjectData || !analysisResults) {
        showError('No analysis results available to generate report');
        return;
    }

    showLoadingSpinner(true);

    fetch(`/api/report/${currentSubjectData.id}`, {
        method: 'POST'
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to generate report');
        }
        return response.blob();
    })
    .then(blob => {
        // Create download link
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `report_${currentSubjectData.id}.pdf`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        showMessage('Report generated successfully');
        showLoadingSpinner(false);
    })
    .catch(error => {
        console.error('Error generating report:', error);
        showError('Failed to generate report');
        showLoadingSpinner(false);
    });
}

/**
 * Update UI with subject information
 * @param {Object} data - Subject data
 */
function updateSubjectInfo(data) {
    const subjectInfoContainer = document.getElementById('subject-info-container');
    if (!subjectInfoContainer) return;

    let html = '<h3>Subject Information</h3>';
    html += `<p><strong>ID:</strong> ${data.id}</p>`;

    if (data.demographic) {
        html += `<p><strong>Age:</strong> ${data.demographic.age}</p>`;
        html += `<p><strong>Sex:</strong> ${data.demographic.sex}</p>`;
        html += `<p><strong>Group:</strong> ${data.demographic.group}</p>`;
    }

    if (data.clinical) {
        html += '<h4>Clinical Data</h4>';
        html += '<table class="clinical-table">';
        html += '<tr><th>Measure</th><th>Value</th></tr>';

        for (const [measure, value] of Object.entries(data.clinical)) {
            html += `<tr><td>${measure}</td><td>${value}</td></tr>`;
        }

        html += '</table>';
    }

    subjectInfoContainer.innerHTML = html;
}

/**
 * Reset all form fields
 */
function resetForms() {
    const forms = document.querySelectorAll('form');
    forms.forEach(form => form.reset());
}

/**
 * Show a loading spinner
 * @param {boolean} show - Whether to show or hide the spinner
 */
function showLoadingSpinner(show) {
    const spinner = document.getElementById('loading-spinner');
    if (spinner) {
        spinner.style.display = show ? 'flex' : 'none';
    }
}

/**
 * Show an error message to the user
 * @param {string} message - Error message to display
 */
function showError(message) {
    const errorContainer = document.getElementById('error-container');
    if (errorContainer) {
        errorContainer.textContent = message;
        errorContainer.style.display = 'block';

        // Hide after 5 seconds
        setTimeout(() => {
            errorContainer.style.display = 'none';
        }, 5000);
    }
}

/**
 * Show a success message to the user
 * @param {string} message - Message to display
 */
function showMessage(message) {
    const messageContainer = document.getElementById('message-container');
    const messageElement = document.createElement('div');
    messageElement.className = 'alert alert-info alert-dismissible fade show';
    messageElement.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    messageContainer.appendChild(messageElement);

    // Auto dismiss after 5 seconds
    setTimeout(() => {
        messageElement.classList.remove('show');
        setTimeout(() => messageElement.remove(), 500);
    }, 5000);
}

// Brain visualization using Papaya.js viewer
let viewer = null;

function initializeViewer() {
    if (document.getElementById('brain-viewer')) {
        const params = {
            worldSpace: true,
            expandable: true,
            kioskMode: false,
            allowScroll: true,
            showControls: true,
            showControlBar: true,
            showImageButtons: true
        };

        papaya.Container.startPapaya();
        viewer = papaya.Container.instances[0];
        showMessage("Brain viewer initialized successfully");
    }
}

function loadNiftiImage(imageUrl) {
    if (!viewer) {
        initializeViewer();
    }

    showMessage(`Loading brain scan: ${imageUrl}`);

    try {
        papaya.Container.addImage(0, imageUrl);
        document.getElementById('segmentation-panel').classList.remove('d-none');
    } catch (error) {
        showMessage(`Error loading image: ${error}`);
    }
}

// API interactions
function fetchPatientList() {
    showMessage("Fetching patient list...");

    fetch('/api/patients')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            populatePatientDropdown(data);
            showMessage(`Loaded ${data.length} patients`);
        })
        .catch(error => {
            showMessage(`Error fetching patient list: ${error.message}`);
        });
}

function populatePatientDropdown(patients) {
    const dropdown = document.getElementById('patient-selector');
    dropdown.innerHTML = '<option value="">Select a patient</option>';

    patients.forEach(patient => {
        const option = document.createElement('option');
        option.value = patient.id;
        option.textContent = `${patient.id} - ${patient.group} (${patient.age}y ${patient.sex})`;
        dropdown.appendChild(option);
    });
}

function loadPatientData(patientId) {
    if (!patientId) return;

    showMessage(`Loading data for patient ${patientId}...`);

    fetch(`/api/patients/${patientId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            displayPatientDetails(data);
            if (data.mri_path) {
                loadNiftiImage(data.mri_path);
            }
            fetchClinicalScores(patientId);
        })
        .catch(error => {
            showMessage(`Error loading patient data: ${error.message}`);
        });
}

function displayPatientDetails(patient) {
    const detailsContainer = document.getElementById('patient-details');
    detailsContainer.innerHTML = `
        <h4>Patient ${patient.id}</h4>
        <p><strong>Group:</strong> ${patient.group}</p>
        <p><strong>Age:</strong> ${patient.age}</p>
        <p><strong>Sex:</strong> ${patient.sex}</p>
        <p><strong>Education:</strong> ${patient.education} years</p>
    `;
    detailsContainer.classList.remove('d-none');
}

function fetchClinicalScores(patientId) {
    fetch(`/api/patients/${patientId}/clinical`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            displayClinicalScores(data);
            createClinicalCharts(data);
        })
        .catch(error => {
            showMessage(`Error fetching clinical scores: ${error.message}`);
        });
}

function displayClinicalScores(data) {
    const scoresContainer = document.getElementById('clinical-scores');

    let scoreHtml = '<h4>Clinical Assessments</h4><table class="table table-striped"><tbody>';

    if (data.mmse) {
        scoreHtml += `<tr><td>MMSE Score</td><td>${data.mmse.total}</td><td>${interpretMMSE(data.mmse.total)}</td></tr>`;
    }

    if (data.cdr) {
        scoreHtml += `<tr><td>CDR Score</td><td>${data.cdr.global}</td><td>${interpretCDR(data.cdr.global)}</td></tr>`;
    }

    if (data.gdscale) {
        scoreHtml += `<tr><td>Geriatric Depression Scale</td><td>${data.gdscale.total}</td><td>${interpretGDS(data.gdscale.total)}</td></tr>`;
    }

    if (data.adas) {
        scoreHtml += `<tr><td>ADAS-Cog</td><td>${data.adas.total}</td><td>${interpretADAS(data.adas.total)}</td></tr>`;
    }

    scoreHtml += '</tbody></table>';
    scoresContainer.innerHTML = scoreHtml;
    scoresContainer.classList.remove('d-none');
}

// Interpretation helpers
function interpretMMSE(score) {
    if (score >= 24) return 'Normal';
    if (score >= 19) return 'Mild cognitive impairment';
    if (score >= 10) return 'Moderate cognitive impairment';
    return 'Severe cognitive impairment';
}

function interpretCDR(score) {
    const cdrMap = {
        0: 'Normal',
        0.5: 'Very mild dementia',
        1: 'Mild dementia',
        2: 'Moderate dementia',
        3: 'Severe dementia'
    };
    return cdrMap[score] || 'Unknown';
}

function interpretGDS(score) {
    if (score <= 9) return 'Normal';
    if (score <= 19) return 'Mild depression';
    return 'Severe depression';
}

function interpretADAS(score) {
    if (score <= 10) return 'Normal cognition';
    if (score <= 18) return 'Mild cognitive impairment';
    return 'Significant cognitive impairment';
}

// Data visualization with Chart.js
function createClinicalCharts(data) {
    const chartsContainer = document.getElementById('clinical-charts');
    chartsContainer.innerHTML = '<canvas id="radar-chart"></canvas><canvas id="volume-chart"></canvas>';

    // Create radar chart for clinical scores
    createRadarChart(data);

    // Create bar chart for brain volumes
    if (data.volumes) {
        createVolumeChart(data.volumes);
    }

    chartsContainer.classList.remove('d-none');
}

function createRadarChart(data) {
    const ctx = document.getElementById('radar-chart').getContext('2d');

    // Normalize scores for radar chart (higher is better)
    let mmseNormalized = data.mmse ? (data.mmse.total / 30) * 100 : 0;
    let adasNormalized = data.adas ? 100 - ((data.adas.total / 70) * 100) : 0;
    let gdsNormalized = data.gdscale ? 100 - ((data.gdscale.total / 30) * 100) : 0;

    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['MMSE', 'ADAS-Cog', 'GDS', 'Memory', 'Executive Function'],
            datasets: [{
                label: 'Patient Scores',
                data: [
                    mmseNormalized,
                    adasNormalized,
                    gdsNormalized,
                    data.neurobat?.memory || 0,
                    data.neurobat?.executive || 0
                ],
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }, {
                label: 'Normal Range',
                data: [80, 80, 80, 80, 80],
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1,
                borderDash: [5, 5]
            }]
        },
        options: {
            responsive: true,
            scales: {
                r: {
                    angleLines: {
                        display: true
                    },
                    suggestedMin: 0,
                    suggestedMax: 100
                }
            }
        }
    });
}

function createVolumeChart(volumes) {
    const ctx = document.getElementById('volume-chart').getContext('2d');

    // Get volume keys and values
    const regions = Object.keys(volumes);
    const values = Object.values(volumes);

    // Create normalized percentile values (if available)
    const percentiles = regions.map((region) => {
        return volumes[`${region}_percentile`] || null;
    });

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: regions,
            datasets: [{
                label: 'Volume (mm³)',
                data: values,
                backgroundColor: 'rgba(153, 102, 255, 0.5)',
                borderColor: 'rgba(153, 102, 255, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            indexAxis: 'y',
            scales: {
                x: {
                    beginAtZero: true
                }
            }
        }
    });
}

// Segmentation and abnormality detection
function runSegmentation() {
    const patientId = document.getElementById('patient-selector').value;
    if (!patientId) {
        showMessage("Please select a patient first");
        return;
    }

    showMessage("Running brain segmentation...");
    document.getElementById('segmentation-btn').disabled = true;

    fetch(`/api/segment/${patientId}`, {
        method: 'POST'
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            showMessage("Segmentation complete");
            loadSegmentationResult(data);
            document.getElementById('segmentation-btn').disabled = false;
        })
        .catch(error => {
            showMessage(`Error during segmentation: ${error.message}`);
            document.getElementById('segmentation-btn').disabled = false;
        });
}

function loadSegmentationResult(data) {
    // Update the viewer with segmentation overlay
    if (data.segmentation_path) {
        try {
            papaya.Container.addImage(0, data.segmentation_path);
            showMessage("Segmentation overlay added");
        } catch (error) {
            showMessage(`Error loading segmentation: ${error}`);
        }
    }

    // Display abnormality results
    if (data.abnormalities) {
        displayAbnormalities(data.abnormalities);
    }

    // Show volume results
    if (data.volumes) {
        createVolumeChart(data.volumes);
    }

    // Enable report generation
    document.getElementById('generate-report-btn').classList.remove('d-none');
}

function displayAbnormalities(abnormalities) {
    const container = document.getElementById('abnormalities');
    container.innerHTML = '<h4>Detected Abnormalities</h4>';

    if (abnormalities.length === 0) {
        container.innerHTML += '<div class="alert alert-success">No significant abnormalities detected</div>';
    } else {
        let abnormalityHtml = '<div class="list-group">';

        abnormalities.forEach(item => {
            const severityClass = getSeverityClass(item.severity);
            abnormalityHtml += `
                <div class="list-group-item ${severityClass}">
                    <div class="d-flex w-100 justify-content-between">
                        <h5 class="mb-1">${item.type}</h5>
                        <small>Confidence: ${(item.confidence * 100).toFixed(1)}%</small>
                    </div>
                    <p class="mb-1">Region: ${item.region}</p>
                    <small>Volume: ${item.volume} mm³</small>
                </div>
            `;
        });

        abnormalityHtml += '</div>';
        container.innerHTML += abnormalityHtml;
    }

    container.classList.remove('d-none');
}

function getSeverityClass(severity) {
    switch(severity) {
        case 'high':
            return 'list-group-item-danger';
        case 'medium':
            return 'list-group-item-warning';
        case 'low':
            return 'list-group-item-info';
        default:
            return '';
    }
}

function generateReport() {
    const patientId = document.getElementById('patient-selector').value;
    if (!patientId) {
        showMessage("Please select a patient first");
        return;
    }

    showMessage("Generating clinical report...");

    fetch(`/api/report/${patientId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.blob();
        })
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `brain_report_${patientId}.pdf`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            showMessage("Report downloaded successfully");
        })
        .catch(error => {
            showMessage(`Error generating report: ${error.message}`);
        });
}

// Statistical analysis
function showStatisticalAnalysis() {
    showMessage("Fetching statistical analysis...");

    fetch('/api/statistics')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            displayStatistics(data);
        })
        .catch(error => {
            showMessage(`Error fetching statistics: ${error.message}`);
        });
}

function displayStatistics(data) {
    const container = document.getElementById('statistics-container');
    container.innerHTML = '<h3>Statistical Analysis</h3>';

    // Group comparisons
    if (data.group_comparisons) {
        container.innerHTML += '<h4>Group Comparisons</h4>';
        createGroupComparisonCharts(data.group_comparisons);
    }

    // Correlation analysis
    if (data.correlations) {
        container.innerHTML += '<h4>Brain-Behavior Correlations</h4>';
        createCorrelationCharts(data.correlations);
    }

    container.classList.remove('d-none');
}

function createGroupComparisonCharts(data) {
    const container = document.getElementById('statistics-container');

    // Create canvas for each region
    data.forEach((region, index) => {
        const canvasId = `group-chart-${index}`;
        const div = document.createElement('div');
        div.className = 'mb-4';
        div.innerHTML = `
            <h5>${region.name}</h5>
            <canvas id="${canvasId}"></canvas>
        `;
        container.appendChild(div);

        // Create the chart
        const ctx = document.getElementById(canvasId).getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['CN', 'MCI', 'AD'],
                datasets: [{
                    label: 'Mean Volume (mm³)',
                    data: [region.cn_mean, region.mci_mean, region.ad_mean],
                    backgroundColor: [
                        'rgba(75, 192, 192, 0.5)',
                        'rgba(255, 205, 86, 0.5)',
                        'rgba(255, 99, 132, 0.5)'
                    ],
                    borderColor: [
                        'rgba(75, 192, 192, 1)',
                        'rgba(255, 205, 86, 1)',
                        'rgba(255, 99, 132, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            afterLabel: function(context) {
                                const group = context.label;
                                const i = context.dataIndex;
                                const std = [region.cn_std, region.mci_std, region.ad_std][i];
                                const pvalue = [region.cn_pvalue, region.mci_pvalue, region.ad_pvalue][i];

                                return [
                                    `SD: ${std.toFixed(2)}`,
                                    `p-value: ${pvalue < 0.001 ? '<0.001' : pvalue.toFixed(3)}`
                                ];
                            }
                        }
                    }
                }
            }
        });
    });
}

function createCorrelationCharts(data) {
    const container = document.getElementById('statistics-container');

    // Create canvas for each correlation
    data.forEach((correlation, index) => {
        const canvasId = `correlation-chart-${index}`;
        const div = document.createElement('div');
        div.className = 'mb-4';
        div.innerHTML = `
            <h5>${correlation.brain_region} vs. ${correlation.clinical_measure}</h5>
            <p>Correlation: r = ${correlation.r.toFixed(3)}, p = ${correlation.p < 0.001 ? '<0.001' : correlation.p.toFixed(3)}</p>
            <canvas id="${canvasId}"></canvas>
        `;
        container.appendChild(div);

        // Prepare scatter plot data
        const scatterData = correlation.data.map(point => {
            return {
                x: point.x,
                y: point.y,
                group: point.group
            };
        });

        // Create different datasets for each group
        const cnData = scatterData.filter(point => point.group === 'CN');
        const mciData = scatterData.filter(point => point.group === 'MCI');
        const adData = scatterData.filter(point => point.group === 'AD');

        // Create the chart
        const ctx = document.getElementById(canvasId).getContext('2d');
        new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Control Normal',
                    data: cnData,
                    backgroundColor: 'rgba(75, 192, 192, 0.5)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }, {
                    label: 'MCI',
                    data: mciData,
                    backgroundColor: 'rgba(255, 205, 86, 0.5)',
                    borderColor: 'rgba(255, 205, 86, 1)',
                    borderWidth: 1
                }, {
                    label: 'AD',
                    data: adData,
                    backgroundColor: 'rgba(255, 99, 132, 0.5)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1
                }, {
                    label: 'Trend Line',
                    data: correlation.trend_line,
                    showLine: true,
                    fill: false,
                    borderColor: 'rgba(128, 128, 128, 1)',
                    borderWidth: 2,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: correlation.x_label
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: correlation.y_label
                        }
                    }
                }
            }
        });
    });
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Initialize the patient selector
    fetchPatientList();

    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Patient selection
    document.getElementById('patient-selector').addEventListener('change', function() {
        const patientId = this.value;
        if (patientId) {
            loadPatientData(patientId);
        }
    });

    // Segmentation button
    const segmentationBtn = document.getElementById('segmentation-btn');
    if (segmentationBtn) {
        segmentationBtn.addEventListener('click', runSegmentation);
    }

    // Report generation button
    const reportBtn = document.getElementById('generate-report-btn');
    if (reportBtn) {
        reportBtn.addEventListener('click', generateReport);
    }

    // Statistical analysis button
    const statsBtn = document.getElementById('statistics-btn');
    if (statsBtn) {
        statsBtn.addEventListener('click', showStatisticalAnalysis);
    }

    // User info
    const userInfo = document.getElementById('user-info');
    if (userInfo) {
        const currentDateTime = '2025-03-31 05:58:03'; // This is from the user's input
        const currentUser = 'KishoreKumarKalli'; // This is from the user's input

        userInfo.textContent = `User: ${currentUser} | Last Updated: ${currentDateTime}`;
    }
});