/**
 * Brain AI - Main JavaScript
 *
 * This file contains the core JavaScript functionality for the Brain AI web application.
 *
 * Version: 1.0.0
 * Last Updated: 2025-03-31
 */

// Wait for the document to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('Brain AI Web Application Initialized');
    initializeComponents();
});

/**
 * Initialize all UI components
 */
function initializeComponents() {
    initializeTooltips();
    initializeImageViewers();
    initializeDataVisualization();
    initializeFormValidation();
    setupEventListeners();

    // Initialize modals if Bootstrap is present
    if (typeof bootstrap !== 'undefined') {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
}

/**
 * Initialize tooltips for better UX
 */
function initializeTooltips() {
    const tooltipElements = document.querySelectorAll('[data-toggle="tooltip"]');
    tooltipElements.forEach(element => {
        new bootstrap.Tooltip(element);
    });
}

/**
 * Set up event listeners for various interactive elements
 */
function setupEventListeners() {
    // File upload preview
    const fileInput = document.getElementById('scan-file-input');
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }

    // Processing form submission
    const processForm = document.getElementById('processing-form');
    if (processForm) {
        processForm.addEventListener('submit', function(e) {
            const loadingSpinner = document.getElementById('processing-spinner');
            if (loadingSpinner) {
                loadingSpinner.classList.remove('d-none');
            }
        });
    }

    // Analysis form
    const analysisForm = document.getElementById('analysis-form');
    if (analysisForm) {
        analysisForm.addEventListener('submit', handleAnalysisSubmit);
    }

    // Slice navigator for 3D scan viewing
    setupSliceNavigator();

    // Add click handler for segmentation legend toggle
    const legendToggle = document.getElementById('toggle-legend');
    if (legendToggle) {
        legendToggle.addEventListener('click', function() {
            const legend = document.getElementById('segmentation-legend');
            if (legend) {
                legend.classList.toggle('d-none');
            }
        });
    }
}

/**
 * Handle file selection for upload preview
 */
function handleFileSelect(event) {
    const fileInput = event.target;
    const fileNameElement = document.getElementById('selected-file-name');
    const filePreviewElement = document.getElementById('file-upload-preview');
    const fileValidFeedback = document.getElementById('file-valid-feedback');

    if (fileNameElement) {
        if (fileInput.files.length > 0) {
            const fileName = fileInput.files[0].name;
            fileNameElement.textContent = fileName;
            fileNameElement.classList.remove('text-muted');
            fileNameElement.classList.add('text-success');

            if (fileValidFeedback) {
                fileValidFeedback.classList.remove('d-none');
            }

            // For NIfTI files we can't show a preview, but we can show a generic brain icon
            if (filePreviewElement) {
                filePreviewElement.innerHTML = '<i class="fas fa-brain fa-5x text-primary"></i>';
                filePreviewElement.classList.remove('d-none');
            }
        } else {
            fileNameElement.textContent = 'No file selected';
            fileNameElement.classList.remove('text-success');
            fileNameElement.classList.add('text-muted');

            if (fileValidFeedback) {
                fileValidFeedback.classList.add('d-none');
            }

            if (filePreviewElement) {
                filePreviewElement.classList.add('d-none');
            }
        }
    }
}

/**
 * Initialize image viewers for brain scan results
 */
function initializeImageViewers() {
    const viewerContainers = document.querySelectorAll('.scan-viewer');

    viewerContainers.forEach(container => {
        const canvas = container.querySelector('canvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const img = new Image();
        img.src = container.dataset.imgSrc;

        img.onload = function() {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);

            // Add zoom functionality
            let scale = 1;
            const zoomIn = container.querySelector('.zoom-in');
            const zoomOut = container.querySelector('.zoom-out');
            const zoomReset = container.querySelector('.zoom-reset');

            if (zoomIn) {
                zoomIn.addEventListener('click', () => {
                    scale *= 1.2;
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.scale(1.2, 1.2);
                    ctx.drawImage(img, 0, 0);
                });
            }

            if (zoomOut) {
                zoomOut.addEventListener('click', () => {
                    scale /= 1.2;
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.scale(0.8, 0.8);
                    ctx.drawImage(img, 0, 0);
                });
            }

            if (zoomReset) {
                zoomReset.addEventListener('click', () => {
                    scale = 1;
                    ctx.setTransform(1, 0, 0, 1, 0, 0);
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0);
                });
            }
        };
    });
}

/**
 * Set up the slice navigator for 3D scan viewing
 */
function setupSliceNavigator() {
    const sliceNavigator = document.getElementById('slice-navigator');
    if (!sliceNavigator) return;

    const sliceDisplay = document.getElementById('current-slice');
    const sliderElement = document.getElementById('slice-slider');
    const viewerElement = document.getElementById('slice-viewer');

    if (!sliderElement || !viewerElement || !sliceDisplay) return;

    // Get total slices from the data attribute
    const totalSlices = parseInt(sliderElement.dataset.totalSlices || 0);
    if (!totalSlices) return;

    // Initialize the slider
    sliderElement.max = totalSlices - 1;
    sliderElement.value = Math.floor(totalSlices / 2); // Start in the middle

    // Initial slice display
    sliceDisplay.textContent = `Slice: ${sliderElement.value + 1} / ${totalSlices}`;

    // Update the image when the slider changes
    sliderElement.addEventListener('input', function() {
        const sliceIndex = parseInt(this.value);
        sliceDisplay.textContent = `Slice: ${sliceIndex + 1} / ${totalSlices}`;

        // The base URL for the slice images
        const baseUrl = viewerElement.dataset.sliceBaseUrl;
        if (baseUrl) {
            // Update the displayed slice image
            viewerElement.src = `${baseUrl}/${sliceIndex}.png`;
        }
    });
}

/**
 * Initialize data visualization components (charts, plots)
 */
function initializeDataVisualization() {
    // Volume charts with Chart.js if available
    initializeVolumeCharts();

    // Initialize comparison charts if on the analysis page
    initializeComparisonCharts();
}

/**
 * Initialize brain region volume charts
 */
function initializeVolumeCharts() {
    const volumeChartCanvas = document.getElementById('volume-chart');
    if (!volumeChartCanvas || typeof Chart === 'undefined') return;

    // Check if we have data
    if (!volumeChartCanvas.dataset.volumes) return;

    try {
        // Parse the volume data from the data attribute
        const volumeData = JSON.parse(volumeChartCanvas.dataset.volumes);
        const labels = Object.keys(volumeData);
        const values = Object.values(volumeData);

        // Create a color array for different brain regions
        const colors = [
            'rgba(255, 99, 132, 0.7)',  // Red
            'rgba(54, 162, 235, 0.7)',   // Blue
            'rgba(255, 206, 86, 0.7)',   // Yellow
            'rgba(75, 192, 192, 0.7)',   // Teal
            'rgba(153, 102, 255, 0.7)',  // Purple
            'rgba(255, 159, 64, 0.7)',   // Orange
            'rgba(199, 199, 199, 0.7)'   // Gray
        ];

        // Create the chart
        new Chart(volumeChartCanvas, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Volume (mm³)',
                    data: values,
                    backgroundColor: colors.slice(0, labels.length),
                    borderColor: colors.slice(0, labels.length).map(color => color.replace('0.7', '1')),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Volume (mm³)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Brain Region'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Brain Region Volumes'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const value = context.raw;
                                return `Volume: ${value.toLocaleString()} mm³`;
                            }
                        }
                    }
                }
            }
        });
    } catch (e) {
        console.error('Error initializing volume chart:', e);
    }
}

/**
 * Initialize comparison charts for group analysis
 */
function initializeComparisonCharts() {
    const comparisonChartCanvas = document.getElementById('comparison-chart');
    if (!comparisonChartCanvas || typeof Chart === 'undefined') return;

    // If we're on the analysis page but don't have data yet, set up the form handler
    if (!comparisonChartCanvas.dataset.chartData) return;

    try {
        // Parse the comparison data
        const chartData = JSON.parse(comparisonChartCanvas.dataset.chartData);

        // Create the comparison chart
        new Chart(comparisonChartCanvas, {
            type: 'bar',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    } catch (e) {
        console.error('Error initializing comparison chart:', e);
    }
}

/**
 * Handle statistical analysis form submission
 */
function handleAnalysisSubmit(event) {
    event.preventDefault();

    const form = event.target;
    const formData = new FormData(form);
    const jsonData = {};

    formData.forEach((value, key) => {
        if (jsonData[key]) {
            if (!Array.isArray(jsonData[key])) {
                jsonData[key] = [jsonData[key]];
            }
            jsonData[key].push(value);
        } else {
            jsonData[key] = value;
        }
    });

    // Convert checkbox groups to arrays
    if (jsonData.clinical_data && !Array.isArray(jsonData.clinical_data)) {
        jsonData.clinical_data = [jsonData.clinical_data];
    }

    if (jsonData.groups && !Array.isArray(jsonData.groups)) {
        jsonData.groups = [jsonData.groups];
    }

    // Show loading indicator
    const resultsContainer = document.getElementById('analysis-results');
    if (resultsContainer) {
        resultsContainer.innerHTML = '<div class="text-center"><div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div><p class="mt-2">Processing analysis...</p></div>';
    }

    // Send the analysis request
    fetch('/api/analysis', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(jsonData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            showAnalysisError(data.error);
            return;
        }

        // Display the analysis results
        displayAnalysisResults(data.results);
    })
    .catch(error => {
        showAnalysisError(`Error: ${error.message}`);
    });
}

/**
 * Display statistical analysis results
 */
function displayAnalysisResults(results) {
    const resultsContainer = document.getElementById('analysis-results');
    if (!resultsContainer) return;

    // Clear previous results
    resultsContainer.innerHTML = '';

    // Create results UI
    const card = document.createElement('div');
    card.className = 'card mt-4';

    const cardHeader = document.createElement('div');
    cardHeader.className = 'card-header bg-success text-white';
    cardHeader.innerHTML = '<h3>Analysis Results</h3>';

    const cardBody = document.createElement('div');
    cardBody.className = 'card-body';

    // Add the results based on the type of analysis
    if (results.type === 'group_comparison') {
        // Group comparison results
        const groupsTable = createGroupComparisonTable(results.data);
        cardBody.appendChild(groupsTable);

        // Add visualization if there are metrics to visualize
        if (results.data && results.data.metrics) {
            const chartContainer = document.createElement('div');
            chartContainer.className = 'mt-4';
            chartContainer.innerHTML = '<h4>Visual Comparison</h4>';

            const canvas = document.createElement('canvas');
            canvas.id = 'group-comparison-chart';
            canvas.height = 300;
            chartContainer.appendChild(canvas);
            cardBody.appendChild(chartContainer);

            // Create chart data
            const chartData = prepareGroupComparisonChartData(results.data);

            // Initialize the chart
            new Chart(canvas, {
                type: 'bar',
                data: chartData,
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Value'
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Group Comparison by Metrics'
                        }
                    }
                }
            });
        }

        // Add statistical significance results
        if (results.statistics) {
            const statsSection = document.createElement('div');
            statsSection.className = 'mt-4';
            statsSection.innerHTML = '<h4>Statistical Analysis</h4>';

            const statsTable = createStatisticsTable(results.statistics);
            statsSection.appendChild(statsTable);
            cardBody.appendChild(statsSection);
        }
    } else if (results.type === 'correlation') {
        // Correlation analysis results
        const correlationSection = document.createElement('div');
        correlationSection.innerHTML = '<h4>Correlation Analysis</h4>';

        // Add correlation matrix
        if (results.correlation_matrix) {
            const matrixTable = createCorrelationMatrixTable(results.correlation_matrix);
            correlationSection.appendChild(matrixTable);
        }

        // Add visualization
        const chartContainer = document.createElement('div');
        chartContainer.className = 'mt-4';
        chartContainer.innerHTML = '<h4>Correlation Plot</h4>';

        const canvas = document.createElement('canvas');
        canvas.id = 'correlation-chart';
        canvas.height = 300;
        chartContainer.appendChild(canvas);

        correlationSection.appendChild(chartContainer);
        cardBody.appendChild(correlationSection);

        // Initialize correlation heatmap
        if (results.correlation_matrix) {
            initializeCorrelationHeatmap('correlation-chart', results.correlation_matrix);
        }
    } else if (results.type === 'longitudinal') {
        // Longitudinal analysis results
        cardBody.innerHTML = '<h4>Longitudinal Analysis</h4>';
        cardBody.innerHTML += createLongitudinalResultsHTML(results.data);
    }

    // Add conclusions if available
    if (results.conclusions) {
        const conclusionsSection = document.createElement('div');
        conclusionsSection.className = 'alert alert-info mt-4';
        conclusionsSection.innerHTML = '<h4>Conclusions</h4>';

        const conclusionsList = document.createElement('ul');
        results.conclusions.forEach(conclusion => {
            const li = document.createElement('li');
            li.textContent = conclusion;
            conclusionsList.appendChild(li);
        });

        conclusionsSection.appendChild(conclusionsList);
        cardBody.appendChild(conclusionsSection);
    }

    card.appendChild(cardHeader);
    card.appendChild(cardBody);
    resultsContainer.appendChild(card);

    // Add download button for results
    const downloadBtn = document.createElement('button');
    downloadBtn.className = 'btn btn-primary mt-3';
    downloadBtn.innerHTML = '<i class="fas fa-download"></i> Download Results as CSV';
    downloadBtn.addEventListener('click', () => {
        downloadResultsAsCSV(results);
    });

    resultsContainer.appendChild(downloadBtn);
}

/**
 * Create a table for group comparison results
 */
function createGroupComparisonTable(data) {
    const table = document.createElement('table');
    table.className = 'table table-striped table-bordered';

    // Create table header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');

    // Add "Metric" header
    const metricHeader = document.createElement('th');
    metricHeader.textContent = 'Metric';
    headerRow.appendChild(metricHeader);

    // Add group headers
    data.groups.forEach(group => {
        const groupHeader = document.createElement('th');
        groupHeader.textContent = group;
        headerRow.appendChild(groupHeader);
    });

    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Create table body
    const tbody = document.createElement('tbody');

    // Add rows for each metric
    Object.keys(data.metrics).forEach(metric => {
        const row = document.createElement('tr');

        // Add metric name
        const metricCell = document.createElement('td');
        metricCell.textContent = metric;
        row.appendChild(metricCell);

        // Add values for each group
        data.groups.forEach(group => {
            const valueCell = document.createElement('td');
            const value = data.metrics[metric][group];

            // Format the value based on type
            if (typeof value === 'number') {
                valueCell.textContent = value.toFixed(4);
            } else {
                valueCell.textContent = value;
            }

            row.appendChild(valueCell);
        });

        tbody.appendChild(row);
    });

    table.appendChild(tbody);
    return table;
}

/**
 * Create a table for statistical significance results
 */
function createStatisticsTable(statistics) {
    const table = document.createElement('table');
    table.className = 'table table-striped table-bordered';

    // Create table header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');

    ['Comparison', 'Test Type', 'p-value', 'Significant'].forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });

    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Create table body
    const tbody = document.createElement('tbody');

    // Add rows for each statistical test
    statistics.forEach(stat => {
        const row = document.createElement('tr');

        // Comparison
        const comparisonCell = document.createElement('td');
        comparisonCell.textContent = stat.comparison;
        row.appendChild(comparisonCell);

        // Test type
        const testTypeCell = document.createElement('td');
        testTypeCell.textContent = stat.test_type;
        row.appendChild(testTypeCell);

        // p-value
        const pValueCell = document.createElement('td');
        pValueCell.textContent = stat.p_value.toFixed(4);
        row.appendChild(pValueCell);

        // Significance
        const sigCell = document.createElement('td');
        const isSignificant = stat.p_value < 0.05;
        sigCell.textContent = isSignificant ? 'Yes (p < 0.05)' : 'No';
        sigCell.className = isSignificant ? 'text-success' : 'text-danger';
        row.appendChild(sigCell);

        tbody.appendChild(row);
    });

    table.appendChild(tbody);
    return table;
}

/**
 * Create a table for correlation matrix
 */
function createCorrelationMatrixTable(correlationMatrix) {
    const table = document.createElement('table');
    table.className = 'table table-sm