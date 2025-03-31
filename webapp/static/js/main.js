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
function createCorrelationMatrix(data, labels, containerId) {
    // Clear the container
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    // Create the table
    const table = document.createElement('table');
    table.className = 'correlation-matrix table table-bordered table-sm';

    // Create header row with labels
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');

    // Empty cell for top-left corner
    headerRow.appendChild(document.createElement('th'));

    // Add variable names as column headers
    labels.forEach(label => {
        const th = document.createElement('th');
        th.textContent = label;
        th.scope = 'col';
        headerRow.appendChild(th);
    });

    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Create table body with correlation values
    const tbody = document.createElement('tbody');

    data.forEach((row, rowIndex) => {
        const tr = document.createElement('tr');

        // Add variable name as row header
        const th = document.createElement('th');
        th.textContent = labels[rowIndex];
        th.scope = 'row';
        tr.appendChild(th);

        // Add correlation values
        row.forEach((value, colIndex) => {
            const td = document.createElement('td');

            // Format the value
            const formattedValue = value.toFixed(2);
            td.textContent = formattedValue;

            // Color coding based on correlation value
            if (rowIndex !== colIndex) { // Skip diagonal (self-correlation)
                const absValue = Math.abs(value);
                let bgColor;

                if (value > 0) {
                    // Positive correlation (blue scale)
                    const intensity = Math.min(absValue * 100, 100);
                    bgColor = `rgba(0, 123, 255, ${intensity/100})`;
                } else {
                    // Negative correlation (red scale)
                    const intensity = Math.min(absValue * 100, 100);
                    bgColor = `rgba(220, 53, 69, ${intensity/100})`;
                }

                td.style.backgroundColor = bgColor;

                // Set text color based on background intensity for readability
                if (absValue > 0.5) {
                    td.style.color = 'white';
                }
            } else {
                // Diagonal elements (self-correlation = 1.0)
                td.style.backgroundColor = '#f8f9fa';
            }

            tr.appendChild(td);
        });

        tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    container.appendChild(table);

    // Add a legend for the correlation matrix
    const legend = document.createElement('div');
    legend.className = 'correlation-legend mt-2 d-flex justify-content-center';

    const legendItems = [
        { color: 'rgba(220, 53, 69, 1)', label: 'Perfect negative (-1.0)' },
        { color: 'rgba(220, 53, 69, 0.5)', label: 'Moderate negative (-0.5)' },
        { color: '#f8f9fa', label: 'No correlation (0.0)' },
        { color: 'rgba(0, 123, 255, 0.5)', label: 'Moderate positive (0.5)' },
        { color: 'rgba(0, 123, 255, 1)', label: 'Perfect positive (1.0)' }
    ];

    legendItems.forEach(item => {
        const legendItem = document.createElement('div');
        legendItem.className = 'mx-2 d-flex align-items-center';

        const colorBox = document.createElement('div');
        colorBox.className = 'correlation-color-box';
        colorBox.style.width = '20px';
        colorBox.style.height = '20px';
        colorBox.style.backgroundColor = item.color;
        colorBox.style.display = 'inline-block';
        colorBox.style.marginRight = '5px';
        colorBox.style.border = '1px solid #dee2e6';

        const label = document.createElement('span');
        label.textContent = item.label;
        label.className = 'small';

        legendItem.appendChild(colorBox);
        legendItem.appendChild(label);
        legend.appendChild(legendItem);
    });

    container.appendChild(legend);
}

/**
 * Create a heatmap visualization for brain region volumes
 * @param {Object} volumeData - The volume data with region names as keys and volumes as values
 * @param {string} containerId - The ID of the container element
 */
function createVolumeHeatmap(volumeData, containerId) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    // Extract regions and volumes
    const regions = Object.keys(volumeData);
    const volumes = Object.values(volumeData);

    // Find the maximum volume for scaling
    const maxVolume = Math.max(...volumes);

    // Create the heatmap container
    const heatmapContainer = document.createElement('div');
    heatmapContainer.className = 'volume-heatmap d-flex flex-wrap justify-content-center';

    // Create heatmap cells
    regions.forEach((region, index) => {
        const cell = document.createElement('div');
        cell.className = 'volume-cell m-1 p-2 text-center';

        // Calculate relative size and color intensity based on volume
        const relativeSize = 50 + (volumes[index] / maxVolume) * 100; // 50px to 150px
        const intensity = 0.3 + (volumes[index] / maxVolume) * 0.7; // 0.3 to 1.0 opacity

        cell.style.width = `${relativeSize}px`;
        cell.style.height = `${relativeSize}px`;
        cell.style.backgroundColor = `rgba(0, 123, 255, ${intensity})`;
        cell.style.color = intensity > 0.6 ? 'white' : 'black';
        cell.style.display = 'flex';
        cell.style.flexDirection = 'column';
        cell.style.justifyContent = 'center';
        cell.style.borderRadius = '4px';

        // Region name and volume
        const regionName = document.createElement('div');
        regionName.textContent = region;
        regionName.className = 'small font-weight-bold';

        const volumeText = document.createElement('div');
        volumeText.textContent = `${Math.round(volumes[index])} mm³`;
        volumeText.className = 'smaller';

        cell.appendChild(regionName);
        cell.appendChild(volumeText);
        cell.title = `${region}: ${Math.round(volumes[index])} mm³`;

        heatmapContainer.appendChild(cell);
    });

    container.appendChild(heatmapContainer);
}

/**
 * Create a grouped bar chart for comparison between groups
 * @param {Object} data - The data object with groups and values
 * @param {string} containerId - The ID of the container element
 * @param {string} title - Chart title
 */
function createGroupedBarChart(data, containerId, title) {
    const container = document.getElementById(containerId);

    // Create canvas for chart
    const canvas = document.createElement('canvas');
    canvas.id = `chart-${containerId}`;
    container.innerHTML = '';
    container.appendChild(canvas);

    // Prepare data for Chart.js
    const labels = Object.keys(data[Object.keys(data)[0]]);
    const datasets = [];
    const colorPalette = [
        'rgba(0, 123, 255, 0.7)',   // blue
        'rgba(220, 53, 69, 0.7)',    // red
        'rgba(40, 167, 69, 0.7)',    // green
        'rgba(255, 193, 7, 0.7)',    // yellow
        'rgba(111, 66, 193, 0.7)',   // purple
        'rgba(23, 162, 184, 0.7)',   // cyan
    ];

    // Create datasets for each group
    Object.keys(data).forEach((group, index) => {
        datasets.push({
            label: group,
            data: Object.values(data[group]),
            backgroundColor: colorPalette[index % colorPalette.length],
            borderColor: colorPalette[index % colorPalette.length].replace('0.7', '1'),
            borderWidth: 1
        });
    });

    // Create chart
    new Chart(canvas, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: title,
                    font: {
                        size: 16
                    }
                },
                legend: {
                    position: 'top',
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Value'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Metric'
                    }
                }
            }
        }
    });
}

/**
 * Create a scatter plot for two variables
 * @param {Array} data - Array of data points with x and y values
 * @param {string} xLabel - Label for x-axis
 * @param {string} yLabel - Label for y-axis
 * @param {string} containerId - The ID of the container element
 * @param {Array} groupLabels - Optional array of group labels for coloring points
 */
function createScatterPlot(data, xLabel, yLabel, containerId, groupLabels = null) {
    const container = document.getElementById(containerId);

    // Create canvas for chart
    const canvas = document.createElement('canvas');
    canvas.id = `scatter-${containerId}`;
    container.innerHTML = '';
    container.appendChild(canvas);

    // Extract x and y values
    const xValues = data.map(point => point.x);
    const yValues = data.map(point => point.y);

    // Prepare datasets
    let datasets;

    if (groupLabels) {
        // Group data points by label
        const groupedData = {};
        data.forEach((point, index) => {
            const label = groupLabels[index];
            if (!groupedData[label]) {
                groupedData[label] = [];
            }
            groupedData[label].push(point);
        });

        // Color palette for groups
        const colorPalette = [
            'rgba(0, 123, 255, 0.7)',   // blue
            'rgba(220, 53, 69, 0.7)',   // red
            'rgba(40, 167, 69, 0.7)',   // green
            'rgba(255, 193, 7, 0.7)',   // yellow
            'rgba(111, 66, 193, 0.7)',  // purple
        ];

        // Create a dataset for each group
        datasets = Object.keys(groupedData).map((label, index) => {
            const groupData = groupedData[label];
            return {
                label: label,
                data: groupData,
                backgroundColor: colorPalette[index % colorPalette.length],
                borderColor: colorPalette[index % colorPalette.length].replace('0.7', '1'),
                borderWidth: 1,
                pointRadius: 6,
                pointHoverRadius: 8
            };
        });
    } else {
        // Single dataset for all points
        datasets = [{
            data: data,
            backgroundColor: 'rgba(0, 123, 255, 0.7)',
            borderColor: 'rgba(0, 123, 255, 1)',
            borderWidth: 1,
            pointRadius: 6,
            pointHoverRadius: 8
        }];
    }

    // Calculate regression line if no groups
    let regressionLine = null;
    if (!groupLabels) {
        regressionLine = calculateRegressionLine(xValues, yValues);
    }

    // Create the chart
    const chart = new Chart(canvas, {
        type: 'scatter',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: `${yLabel} vs ${xLabel}`,
                    font: {
                        size: 16
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const point = context.raw;
                            return `${xLabel}: ${point.x.toFixed(2)}, ${yLabel}: ${point.y.toFixed(2)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: xLabel
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: yLabel
                    }
                }
            }
        }
    });

    // Add regression line if calculated
    if (regressionLine) {
        const minX = Math.min(...xValues);
        const maxX = Math.max(...xValues);

        const lineStart = { x: minX, y: regressionLine.slope * minX + regressionLine.intercept };
        const lineEnd = { x: maxX, y: regressionLine.slope * maxX + regressionLine.intercept };

        chart.data.datasets.push({
            type: 'line',
            label: 'Regression Line',
            data: [lineStart, lineEnd],
            fill: false,
            borderColor: 'rgba(255, 99, 132, 1)',
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 0
        });

        // Add R² value
        const rSquared = regressionLine.r2;
        const annotation = document.createElement('div');
        annotation.className = 'regression-stats mt-2 text-center';
        annotation.innerHTML = `<strong>R²:</strong> ${rSquared.toFixed(3)}`;
        container.appendChild(annotation);

        chart.update();
    }
}

/**
 * Calculate linear regression line parameters
 * @param {Array} xValues - Array of x values
 * @param {Array} yValues - Array of y values
 * @returns {Object} Object containing slope, intercept and R²
 */
function calculateRegressionLine(xValues, yValues) {
    const n = xValues.length;

    // Calculate means
    const xMean = xValues.reduce((sum, val) => sum + val, 0) / n;
    const yMean = yValues.reduce((sum, val) => sum + val, 0) / n;

    // Calculate slope and intercept
    let numerator = 0;
    let denominator = 0;

    for (let i = 0; i < n; i++) {
        numerator += (xValues[i] - xMean) * (yValues[i] - yMean);
        denominator += Math.pow(xValues[i] - xMean, 2);
    }

    const slope = numerator / denominator;
    const intercept = yMean - slope * xMean;

    // Calculate R-squared
    let totalSumSquares = 0;
    let residualSumSquares = 0;

    for (let i = 0; i < n; i++) {
        const predictedY = slope * xValues[i] + intercept;
        totalSumSquares += Math.pow(yValues[i] - yMean, 2);
        residualSumSquares += Math.pow(yValues[i] - predictedY, 2);
    }

    const r2 = 1 - (residualSumSquares / totalSumSquares);

    return { slope, intercept, r2 };
}

/**
 * Create a box plot for comparing distributions
 * @param {Object} data - Object with group names as keys and arrays of values as values
 * @param {string} containerId - The ID of the container element
 * @param {string} title - Chart title
 * @param {string} yAxisLabel - Label for y-axis
 */
function createBoxPlot(data, containerId, title, yAxisLabel) {
    const container = document.getElementById(containerId);

    // Create canvas for chart
    const canvas = document.createElement('canvas');
    canvas.id = `boxplot-${containerId}`;
    container.innerHTML = '';
    container.appendChild(canvas);

    // Prepare data for Chart.js boxplot plugin
    const labels = Object.keys(data);
    const datasets = [{
        label: yAxisLabel,
        backgroundColor: 'rgba(0, 123, 255, 0.5)',
        borderColor: 'rgb(0, 123, 255)',
        borderWidth: 1,
        outlierColor: '#999999',
        padding: 10,
        itemRadius: 3,
        data: []
    }];

    // Calculate statistics for each group
    labels.forEach(group => {
        const values = data[group].sort((a, b) => a - b);
        const min = values[0];
        const max = values[values.length - 1];

        const q1Idx = Math.floor(values.length * 0.25);
        const medianIdx = Math.floor(values.length * 0.5);
        const q3Idx = Math.floor(values.length * 0.75);

        const q1 = values[q1Idx];
        const median = values[medianIdx];
        const q3 = values[q3Idx];

        // Find outliers (values outside 1.5 * IQR)
        const iqr = q3 - q1;
        const upperFence = q3 + 1.5 * iqr;
        const lowerFence = q1 - 1.5 * iqr;

        const filteredValues = values.filter(v => v >= lowerFence && v <= upperFence);
        const filteredMin = filteredValues[0];
        const filteredMax = filteredValues[filteredValues.length - 1];

        const outliers = values.filter(v => v < lowerFence || v > upperFence);

        // Format data for Chart.js
        datasets[0].data.push({
            min: filteredMin,
            q1: q1,
            median: median,
            q3: q3,
            max: filteredMax,
            outliers: outliers
        });
    });

    // Create box plot
    new Chart(canvas, {
        type: 'boxplot',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: title,
                    font: {
                        size: 16
                    }
                },
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: yAxisLabel
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Group'
                    }
                }
            }
        }
    });

    // Calculate statistical significance
    if (labels.length > 1) {
        calculateAndDisplayStatistics(data, container);
    }
}
function calculateAndDisplayStatistics(data, container) {
    // Clear the container
    container.empty();

    // Check if data is empty
    if (!data || data.length === 0) {
        container.append('<div class="alert alert-warning">No data available for analysis</div>');
        return;
    }

    // Identify numeric columns for statistical analysis
    const numericColumns = [];
    const categoricalColumns = [];

    // Determine column types from the first data entry
    Object.keys(data[0]).forEach(key => {
        if (!isNaN(parseFloat(data[0][key])) && data[0][key] !== '') {
            numericColumns.push(key);
        } else {
            categoricalColumns.push(key);
        }
    });

    // Skip ID columns and date columns for statistical analysis
    const skipColumns = ['ID', 'PTID', 'RID', 'VISCODE', 'SITEID', 'Date', 'EXAMDATE'];
    const filteredNumericColumns = numericColumns.filter(col => !skipColumns.some(skip => col.toUpperCase().includes(skip)));

    // Create a card for basic statistics
    const statsCard = $(`
        <div class="card mb-4">
            <div class="card-header bg-primary text-white">
                <h5 class="mb-0">Basic Statistics</h5>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-bordered table-striped table-sm">
                        <thead>
                            <tr>
                                <th>Measure</th>
                                <th>Mean</th>
                                <th>Median</th>
                                <th>Std Dev</th>
                                <th>Min</th>
                                <th>Max</th>
                            </tr>
                        </thead>
                        <tbody id="statsTableBody"></tbody>
                    </table>
                </div>
            </div>
        </div>
    `);

    container.append(statsCard);

    // Calculate statistics for each numeric column
    filteredNumericColumns.forEach(column => {
        // Extract values for this column, filtering out missing values
        const values = data.map(row => parseFloat(row[column]))
                           .filter(val => !isNaN(val));

        if (values.length === 0) return;

        // Calculate statistics
        const mean = values.reduce((sum, val) => sum + val, 0) / values.length;

        // Sort values for median and percentiles
        const sortedValues = [...values].sort((a, b) => a - b);
        const median = sortedValues.length % 2 === 0 ?
            (sortedValues[sortedValues.length / 2 - 1] + sortedValues[sortedValues.length / 2]) / 2 :
            sortedValues[Math.floor(sortedValues.length / 2)];

        // Calculate standard deviation
        const sumSquaredDiffs = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0);
        const stdDev = Math.sqrt(sumSquaredDiffs / values.length);

        // Min and Max
        const min = Math.min(...values);
        const max = Math.max(...values);

        // Add to statistics table
        const tableRow = `
            <tr>
                <td>${column}</td>
                <td>${mean.toFixed(2)}</td>
                <td>${median.toFixed(2)}</td>
                <td>${stdDev.toFixed(2)}</td>
                <td>${min.toFixed(2)}</td>
                <td>${max.toFixed(2)}</td>
            </tr>
        `;
        $('#statsTableBody').append(tableRow);
    });

    // Create visualizations for each numeric column
    const visualizationsCard = $(`
        <div class="card mb-4">
            <div class="card-header bg-info text-white">
                <h5 class="mb-0">Visualizations</h5>
            </div>
            <div class="card-body">
                <div class="row" id="visualizationsContainer"></div>
            </div>
        </div>
    `);

    container.append(visualizationsCard);

    // Generate visualizations for top numeric variables
    const topVariables = filteredNumericColumns.slice(0, 6); // Limit to top 6 for better UI

    topVariables.forEach((column, index) => {
        // Create a container for this visualization
        const vizContainer = $(`
            <div class="col-md-6 mb-4">
                <div class="card h-100">
                    <div class="card-header">
                        <h6 class="mb-0">${column}</h6>
                    </div>
                    <div class="card-body">
                        <canvas id="chart-${index}" width="100%" height="250"></canvas>
                    </div>
                </div>
            </div>
        `);

        $('#visualizationsContainer').append(vizContainer);

        // Extract valid values for this column
        const values = data.map(row => parseFloat(row[column]))
                           .filter(val => !isNaN(val));

        if (values.length === 0) return;

        // Create histogram data
        const min = Math.min(...values);
        const max = Math.max(...values);
        const binWidth = (max - min) / 10;
        const bins = Array.from({ length: 10 }, (_, i) => min + i * binWidth);

        const counts = Array(10).fill(0);
        values.forEach(val => {
            for (let i = 0; i < bins.length; i++) {
                if (val >= bins[i] && (i === bins.length - 1 || val < bins[i + 1])) {
                    counts[i]++;
                    break;
                }
            }
        });

        // Format bin labels
        const binLabels = bins.map((val, i) =>
            i === bins.length - 1 ? `${val.toFixed(1)}+` : `${val.toFixed(1)}-${(val + binWidth).toFixed(1)}`
        );

        // Create histogram chart
        const ctx = document.getElementById(`chart-${index}`).getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: binLabels,
                datasets: [{
                    label: column,
                    data: counts,
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
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
                            text: 'Frequency'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: column
                        }
                    }
                }
            }
        });
    });

    // Create correlation analysis if we have multiple numeric columns
    if (filteredNumericColumns.length > 1) {
        const correlationCard = $(`
            <div class="card mb-4">
                <div class="card-header bg-success text-white">
                    <h5 class="mb-0">Correlation Analysis</h5>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-12 mb-4">
                            <canvas id="correlationHeatmap" width="100%" height="400"></canvas>
                        </div>
                        <div class="col-md-12">
                            <div class="table-responsive">
                                <table class="table table-bordered table-sm" id="correlationTable">
                                    <thead id="correlationTableHead"></thead>
                                    <tbody id="correlationTableBody"></tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `);

        container.append(correlationCard);

        // Limit to top 10 variables for correlation analysis to maintain readability
        const correlationColumns = filteredNumericColumns.slice(0, 10);

        // Calculate correlation matrix
        const correlationMatrix = calculateCorrelationMatrix(data, correlationColumns);

        // Create correlation table
        const headRow = $('<tr><th>Variable</th></tr>');
        correlationColumns.forEach(col => {
            headRow.append(`<th>${col}</th>`);
        });
        $('#correlationTableHead').append(headRow);

        correlationColumns.forEach((rowCol, i) => {
            const tableRow = $(`<tr><th>${rowCol}</th></tr>`);

            correlationColumns.forEach((colCol, j) => {
                const correlation = correlationMatrix[i][j];

                // Color code by correlation strength
                let cellClass = '';
                if (i !== j) {  // Skip self-correlations
                    if (Math.abs(correlation) > 0.7) cellClass = 'table-danger';
                    else if (Math.abs(correlation) > 0.5) cellClass = 'table-warning';
                    else if (Math.abs(correlation) > 0.3) cellClass = 'table-info';
                    else cellClass = 'table-light';
                }

                tableRow.append(`<td class="${cellClass}">${correlation.toFixed(2)}</td>`);
            });

            $('#correlationTableBody').append(tableRow);
        });

        // Create correlation heatmap
        const ctx = document.getElementById('correlationHeatmap').getContext('2d');

        // Prepare data for heatmap
        const heatmapData = {
            labels: correlationColumns,
            datasets: correlationColumns.map((col, i) => {
                return {
                    label: col,
                    data: correlationMatrix[i],
                    backgroundColor: (ctx) => {
                        const value = correlationMatrix[i][ctx.dataIndex];

                        // Color scale for correlation: blue (negative) to red (positive)
                        if (ctx.dataIndex === i) {
                            return 'rgba(0, 0, 0, 0.1)';  // Self-correlation
                        } else if (value > 0) {
                            return `rgba(255, 0, 0, ${Math.min(Math.abs(value), 1)})`;
                        } else {
                            return `rgba(0, 0, 255, ${Math.min(Math.abs(value), 1)})`;
                        }
                    }
                };
            })
        };

        new Chart(ctx, {
            type: 'matrix',
            data: {
                datasets: [{
                    label: 'Correlation Matrix',
                    data: generateHeatmapData(correlationMatrix, correlationColumns),
                    backgroundColor: (ctx) => {
                        if (ctx.dataset.data[ctx.dataIndex]) {
                            const value = ctx.dataset.data[ctx.dataIndex].v;

                            // Return transparent for cells where i === j (self-correlation)
                            if (ctx.dataset.data[ctx.dataIndex].x === ctx.dataset.data[ctx.dataIndex].y) {
                                return 'rgba(0, 0, 0, 0.1)';
                            }

                            // Color scale: blue (negative) to white (zero) to red (positive)
                            if (value > 0) {
                                return `rgba(255, 0, 0, ${Math.min(Math.abs(value) * 0.8 + 0.2, 1)})`;
                            } else {
                                return `rgba(0, 0, 255, ${Math.min(Math.abs(value) * 0.8 + 0.2, 1)})`;
                            }
                        }
                        return 'rgba(0, 0, 0, 0)';
                    },
                    borderColor: 'white',
                    borderWidth: 1,
                    width: ({ chart }) => (chart.chartArea || {}).width / correlationColumns.length - 1,
                    height: ({ chart }) => (chart.chartArea || {}).height / correlationColumns.length - 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    tooltip: {
                        callbacks: {
                            title: function(context) {
                                const item = context[0];
                                const data = item.dataset.data[item.dataIndex];
                                return `${correlationColumns[data.y]} vs ${correlationColumns[data.x]}`;
                            },
                            label: function(context) {
                                const value = context.dataset.data[context.dataIndex].v;
                                return `Correlation: ${value.toFixed(2)}`;
                            }
                        }
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        type: 'category',
                        labels: correlationColumns,
                        offset: true,
                        ticks: {
                            minRotation: 45,
                            maxRotation: 45
                        },
                        grid: {
                            display: false
                        },
                        title: {
                            display: true,
                            text: 'Variables'
                        }
                    },
                    y: {
                        type: 'category',
                        labels: correlationColumns,
                        offset: true,
                        reverse: true,
                        grid: {
                            display: false
                        },
                        title: {
                            display: true,
                            text: 'Variables'
                        }
                    }
                }
            }
        });
    }

    // Group Comparison (if categorical columns are available)
    if (categoricalColumns.length > 0) {
        const groupComparisonCard = $(`
            <div class="card mb-4">
                <div class="card-header bg-warning text-white">
                    <h5 class="mb-0">Group Comparison</h5>
                </div>
                <div class="card-body">
                    <div class="row mb-3">
                        <div class="col-md-6">
                            <label for="groupBySelect" class="form-label">Group By:</label>
                            <select class="form-select" id="groupBySelect">
                                <option value="">Select a category</option>
                                ${categoricalColumns.map(col => `<option value="${col}">${col}</option>`).join('')}
                            </select>
                        </div>
                        <div class="col-md-6">
                            <label for="compareVarSelect" class="form-label">Compare Variable:</label>
                            <select class="form-select" id="compareVarSelect">
                                <option value="">Select a variable</option>
                                ${filteredNumericColumns.map(col => `<option value="${col}">${col}</option>`).join('')}
                            </select>
                        </div>
                    </div>
                    <div class="d-grid gap-2 col-md-4 mx-auto mb-4">
                        <button class="btn btn-primary" id="runComparisonBtn">Run Comparison</button>
                    </div>
                    <div id="comparisonResults"></div>
                </div>
            </div>
        `);

        container.append(groupComparisonCard);

        // Handle group comparison button click
        $('#runComparisonBtn').click(function() {
            const groupBy = $('#groupBySelect').val();
            const compareVar = $('#compareVarSelect').val();

            if (!groupBy || !compareVar) {
                $('#comparisonResults').html('<div class="alert alert-warning">Please select both a grouping category and a variable to compare.</div>');
                return;
            }

            // Run the comparison
            runGroupComparison(data, groupBy, compareVar);
        });
    }

    // Longitudinal Analysis (if there's a date or visit code column)
    const possibleDateColumns = data[0] ? Object.keys(data[0]).filter(key =>
        key.toUpperCase().includes('DATE') ||
        key.toUpperCase().includes('VISIT') ||
        key.toUpperCase().includes('VISCODE')
    ) : [];

    if (possibleDateColumns.length > 0) {
        const longitudinalCard = $(`
            <div class="card mb-4">
                <div class="card-header bg-secondary text-white">
                    <h5 class="mb-0">Longitudinal Analysis</h5>
                </div>
                <div class="card-body">
                    <div class="row mb-3">
                        <div class="col-md-4">
                            <label for="timeVariableSelect" class="form-label">Time Variable:</label>
                            <select class="form-select" id="timeVariableSelect">
                                <option value="">Select a time variable</option>
                                ${possibleDateColumns.map(col => `<option value="${col}">${col}</option>`).join('')}
                            </select>
                        </div>
                        <div class="col-md-4">
                            <label for="longitudinalVarSelect" class="form-label">Measure:</label>
                            <select class="form-select" id="longitudinalVarSelect">
                                <option value="">Select a variable</option>
                                ${filteredNumericColumns.map(col => `<option value="${col}">${col}</option>`).join('')}
                            </select>
                        </div>
                        <div class="col-md-4">
                            <label for="groupingVarSelect" class="form-label">Group By (Optional):</label>
                            <select class="form-select" id="groupingVarSelect">
                                <option value="">None</option>
                                ${categoricalColumns.map(col => `<option value="${col}">${col}</option>`).join('')}
                            </select>
                        </div>
                    </div>
                    <div class="d-grid gap-2 col-md-4 mx-auto mb-4">
                        <button class="btn btn-primary" id="runLongitudinalBtn">Run Analysis</button>
                    </div>
                    <div id="longitudinalResults"></div>
                </div>
            </div>
        `);

        container.append(longitudinalCard);

        // Handle longitudinal analysis button click
        $('#runLongitudinalBtn').click(function() {
            const timeVar = $('#timeVariableSelect').val();
            const longitudinalVar = $('#longitudinalVarSelect').val();
            const groupingVar = $('#groupingVarSelect').val();

            if (!timeVar || !longitudinalVar) {
                $('#longitudinalResults').html('<div class="alert alert-warning">Please select both a time variable and a measure.</div>');
                return;
            }

            // Run the longitudinal analysis
            runLongitudinalAnalysis(data, timeVar, longitudinalVar, groupingVar);
        });
    }
}

// Helper function to calculate correlation between two arrays
function calculateCorrelation(x, y) {
    const n = x.length;
    if (n !== y.length || n === 0) return 0;

    // Calculate means
    const xMean = x.reduce((sum, val) => sum + val, 0) / n;
    const yMean = y.reduce((sum, val) => sum + val, 0) / n;

    // Calculate numerator and denominators
    let numerator = 0;
    let xDenominator = 0;
    let yDenominator = 0;

    for (let i = 0; i < n; i++) {
        const xDiff = x[i] - xMean;
        const yDiff = y[i] - yMean;

        numerator += xDiff * yDiff;
        xDenominator += xDiff * xDiff;
        yDenominator += yDiff * yDiff;
    }

    // Avoid division by zero
    if (xDenominator === 0 || yDenominator === 0) return 0;

    return numerator / Math.sqrt(xDenominator * yDenominator);
}

// Calculate correlation matrix for multiple variables
function calculateCorrelationMatrix(data, columns) {
    const n = columns.length;
    const matrix = Array(n).fill().map(() => Array(n).fill(0));

    // Extract arrays of values for each column
    const columnValues = columns.map(col => {
        return data.map(row => parseFloat(row[col]))
                  .filter(val => !isNaN(val));
    });

    // Calculate correlations
    for (let i = 0; i < n; i++) {
        for (let j = i; j < n; j++) {
            // Find common indices where both columns have valid values
            const validI = columnValues[i];
            const validJ = columnValues[j];

            // Only calculate if we have enough data points
            if (validI.length >= 3 && validJ.length >= 3) {
                // Calculate correlation
                const correlation = i === j ? 1.0 : calculateCorrelation(validI, validJ);

                // Fill the matrix (symmetric)
                matrix[i][j] = correlation;
                matrix[j][i] = correlation;
            }
        }
    }

    return matrix;
}
function generateHeatmapData(matrix, labels) {
    // Generates data for the heatmap visualization
    const data = [];

    // Create data array for heatmap plotting
    for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix[i].length; j++) {
            data.push({
                x: labels[j],
                y: labels[i],
                z: matrix[i][j],
                text: `${labels[i]}-${labels[j]}: ${matrix[i][j].toFixed(2)}`,
                hoverinfo: 'text'
            });
        }
    }

    return data;
}

function createHeatmap(elementId, data, labels, title) {
    // Create heatmap visualization using Plotly
    const heatmapData = [{
        x: labels,
        y: labels,
        z: data,
        type: 'heatmap',
        colorscale: 'Viridis',
        hoverongaps: false,
        showscale: true,
        colorbar: {
            title: 'Correlation',
            thickness: 20,
            titleside: 'right',
            tickmode: 'array',
            tickvals: [-1, -0.5, 0, 0.5, 1],
            ticktext: ['-1', '-0.5', '0', '0.5', '1']
        }
    }];

    const layout = {
        title: title || 'Correlation Heatmap',
        margin: {
            l: 120,
            r: 50,
            b: 120,
            t: 100,
            pad: 4
        },
        xaxis: {
            title: '',
            automargin: true
        },
        yaxis: {
            title: '',
            automargin: true
        }
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false
    };

    Plotly.newPlot(elementId, heatmapData, layout, config);
}

function createBarChart(elementId, data, labels, title, xLabel, yLabel) {
    // Create bar chart visualization using Plotly
    const trace = {
        x: labels,
        y: data,
        type: 'bar',
        marker: {
            color: 'rgba(50, 98, 235, 0.7)',
            line: {
                color: 'rgba(50, 98, 235, 1.0)',
                width: 1
            }
        }
    };

    const layout = {
        title: title || 'Bar Chart',
        xaxis: {
            title: xLabel || '',
            tickangle: -45,
            automargin: true
        },
        yaxis: {
            title: yLabel || '',
            automargin: true
        },
        margin: {
            b: 150
        }
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false
    };

    Plotly.newPlot(elementId, [trace], layout, config);
}

function createBoxPlot(elementId, data, groups, title, yLabel) {
    // Create box plot visualization using Plotly
    const traces = [];

    for (let i = 0; i < data.length; i++) {
        traces.push({
            y: data[i],
            type: 'box',
            name: groups[i],
            boxpoints: 'outliers',
            jitter: 0.3,
            pointpos: 0,
            marker: {
                color: getBoxPlotColor(i)
            }
        });
    }

    const layout = {
        title: title || 'Box Plot',
        yaxis: {
            title: yLabel || '',
            zeroline: false
        },
        boxmode: 'group'
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false
    };

    Plotly.newPlot(elementId, traces, layout, config);
}

function getBoxPlotColor(index) {
    // Color palette for box plots
    const colors = [
        'rgba(93, 164, 214, 0.7)', // blue
        'rgba(255, 144, 14, 0.7)',  // orange
        'rgba(44, 160, 101, 0.7)',  // green
        'rgba(255, 65, 54, 0.7)',   // red
        'rgba(207, 114, 255, 0.7)', // purple
        'rgba(127, 96, 0, 0.7)',    // brown
        'rgba(255, 140, 184, 0.7)', // pink
        'rgba(79, 90, 117, 0.7)'    // dark blue
    ];

    return colors[index % colors.length];
}

function createScatterPlot(elementId, xData, yData, labels, title, xLabel, yLabel, groups) {
    // Create scatter plot visualization using Plotly
    const traces = [];

    if (groups) {
        // Create a trace for each group
        const uniqueGroups = [...new Set(groups)];

        uniqueGroups.forEach((group, i) => {
            const indices = groups.map((g, idx) => g === group ? idx : -1).filter(idx => idx !== -1);

            traces.push({
                x: indices.map(idx => xData[idx]),
                y: indices.map(idx => yData[idx]),
                text: indices.map(idx => labels ? labels[idx] : ''),
                mode: 'markers',
                type: 'scatter',
                name: group,
                marker: {
                    size: 10,
                    color: getBoxPlotColor(i),
                    line: {
                        width: 1,
                        color: 'white'
                    }
                }
            });
        });
    } else {
        // Single trace for all data
        traces.push({
            x: xData,
            y: yData,
            text: labels,
            mode: 'markers',
            type: 'scatter',
            marker: {
                size: 10,
                color: 'rgba(93, 164, 214, 0.7)',
                line: {
                    width: 1,
                    color: 'white'
                }
            }
        });
    }

    const layout = {
        title: title || 'Scatter Plot',
        xaxis: {
            title: xLabel || '',
            zeroline: true
        },
        yaxis: {
            title: yLabel || '',
            zeroline: true
        },
        hovermode: 'closest'
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false
    };

    Plotly.newPlot(elementId, traces, layout, config);
}

function createViolinPlot(elementId, data, groups, title, yLabel) {
    // Create violin plot visualization using Plotly
    const traces = [];

    for (let i = 0; i < data.length; i++) {
        traces.push({
            y: data[i],
            type: 'violin',
            name: groups[i],
            box: {
                visible: true
            },
            meanline: {
                visible: true
            },
            marker: {
                color: getBoxPlotColor(i)
            }
        });
    }

    const layout = {
        title: title || 'Violin Plot',
        yaxis: {
            title: yLabel || '',
            zeroline: false
        },
        violinmode: 'group'
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false
    };

    Plotly.newPlot(elementId, traces, layout, config);
}

function loadDemographicOverview() {
    // Load demographic overview data from the API
    $.ajax({
        url: '/api/demographic_overview',
        method: 'GET',
        success: function(response) {
            if (response.error) {
                console.error('Error loading demographic data:', response.error);
                $('#demographicOverview').html('<div class="alert alert-danger">Error loading demographic data</div>');
                return;
            }

            displayDemographicOverview(response);
        },
        error: function(error) {
            console.error('Error fetching demographic data:', error);
            $('#demographicOverview').html('<div class="alert alert-danger">Error loading demographic data</div>');
        }
    });
}

function displayDemographicOverview(data) {
    // Display demographic overview data
    $('#demographicOverview').empty();

    const totalSubjects = data.total_subjects;

    // Create container for overview
    const container = $('<div class="row"></div>');

    // Total subjects card
    const totalCard = $(`
        <div class="col-md-4 mb-4">
            <div class="card bg-primary text-white h-100">
                <div class="card-body text-center">
                    <h5 class="card-title">Total Subjects</h5>
                    <p class="display-4">${totalSubjects}</p>
                </div>
            </div>
        </div>
    `);

    // Age distribution card
    let ageCard;
    if (data.age_stats) {
        ageCard = $(`
            <div class="col-md-4 mb-4">
                <div class="card h-100">
                    <div class="card-body">
                        <h5 class="card-title">Age Distribution</h5>
                        <div class="text-center mb-3">
                            <p>Mean: ${data.age_stats.mean.toFixed(1)} years</p>
                            <p>Range: ${data.age_stats.min.toFixed(0)} - ${data.age_stats.max.toFixed(0)} years</p>
                        </div>
                        <div id="ageHistogram" style="height: 200px;"></div>
                    </div>
                </div>
            </div>
        `);
    } else {
        ageCard = $(`
            <div class="col-md-4 mb-4">
                <div class="card h-100">
                    <div class="card-body">
                        <h5 class="card-title">Age Distribution</h5>
                        <p class="text-muted">No age data available</p>
                    </div>
                </div>
            </div>
        `);
    }

    // Gender distribution card
    let genderCard;
    if (data.gender_stats) {
        genderCard = $(`
            <div class="col-md-4 mb-4">
                <div class="card h-100">
                    <div class="card-body">
                        <h5 class="card-title">Gender Distribution</h5>
                        <div id="genderPieChart" style="height: 200px;"></div>
                    </div>
                </div>
            </div>
        `);
    } else {
        genderCard = $(`
            <div class="col-md-4 mb-4">
                <div class="card h-100">
                    <div class="card-body">
                        <h5 class="card-title">Gender Distribution</h5>
                        <p class="text-muted">No gender data available</p>
                    </div>
                </div>
            </div>
        `);
    }

    // Add cards to container
    container.append(totalCard);
    container.append(ageCard);
    container.append(genderCard);

    // Add container to overview
    $('#demographicOverview').append(container);

    // Create age histogram if data available
    if (data.age_stats && data.age_stats.histogram) {
        const histogramData = [{
            x: Array.from({length: data.age_stats.histogram.length}, (_, i) => {
                const binSize = (data.age_stats.max - data.age_stats.min) / 10;
                return (data.age_stats.min + i * binSize + binSize/2).toFixed(0);
            }),
            y: data.age_stats.histogram,
            type: 'bar',
            marker: {
                color: 'rgba(50, 98, 235, 0.7)',
            }
        }];

        const layout = {
            margin: {l: 40, r: 20, t: 20, b: 40},
            xaxis: {title: 'Age (years)'},
            yaxis: {title: 'Count'},
            bargap: 0.05
        };

        Plotly.newPlot('ageHistogram', histogramData, layout, {responsive: true, displayModeBar: false});
    }

    // Create gender pie chart if data available
    if (data.gender_stats) {
        const genderLabels = Object.keys(data.gender_stats.counts);
        const genderValues = Object.values(data.gender_stats.counts);

        const pieData = [{
            labels: genderLabels,
            values: genderValues,
            type: 'pie',
            marker: {
                colors: ['rgba(93, 164, 214, 0.7)', 'rgba(255, 144, 14, 0.7)']
            },
            textinfo: 'label+percent',
            insidetextorientation: 'radial'
        }];

        const layout = {
            margin: {l: 20, r: 20, t: 20, b: 20},
            showlegend: false
        };

        Plotly.newPlot('genderPieChart', pieData, layout, {responsive: true, displayModeBar: false});
    }
}

function initAnalysisForm() {
    // Initialize the analysis form with available data
    $.ajax({
        url: '/health',
        method: 'GET',
        success: function(response) {
            if (!response.clinical_data) {
                $('#analysisForm').html('<div class="alert alert-warning">Clinical data not available</div>');
                return;
            }

            // Populate available clinical data checkboxes
            const clinicalDataContainer = $('#clinicalDataOptions');
            clinicalDataContainer.empty();

            Object.keys(response.clinical_data).forEach(key => {
                if (response.clinical_data[key]) {
                    const checkbox = $(`
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" value="${key}" id="check_${key}">
                            <label class="form-check-label" for="check_${key}">
                                ${key}
                            </label>
                        </div>
                    `);
                    clinicalDataContainer.append(checkbox);
                }
            });
        },
        error: function(error) {
            console.error('Error checking available data:', error);
            $('#analysisForm').html('<div class="alert alert-danger">Error checking available data</div>');
        }
    });

    // Set up form submission
    $('#runAnalysisBtn').click(function() {
        performAnalysis();
    });
}

function performAnalysis() {
    // Get selected analysis options
    const analysisType = $('#analysisType').val();

    // Get selected clinical data sources
    const clinicalData = [];
    $('#clinicalDataOptions input:checked').each(function() {
        clinicalData.push($(this).val());
    });

    if (clinicalData.length === 0) {
        alert('Please select at least one clinical data source');
        return;
    }

    // Get groups if applicable
    const groups = $('#comparisonGroups').val().split(',').map(g => g.trim()).filter(g => g);

    // Show loading spinner
    $('#analysisResults').html(`
        <div class="text-center my-5">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p>Performing analysis...</p>
        </div>
    `);

    // Call API to perform analysis
    $.ajax({
        url: '/api/analysis',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            analysis_type: analysisType,
            clinical_data: clinicalData,
            groups: groups
        }),
        success: function(response) {
            if (response.error) {
                $('#analysisResults').html(`<div class="alert alert-danger">${response.error}</div>`);
                return;
            }

            displayAnalysisResults(response.results, analysisType);
        },
        error: function(error) {
            console.error('Error performing analysis:', error);
            $('#analysisResults').html('<div class="alert alert-danger">Error performing analysis</div>');
        }
    });
}

function displayAnalysisResults(results, analysisType) {
    // Display the analysis results
    const container = $('#analysisResults');
    container.empty();

    // Add heading
    container.append(`<h3 class="mb-4">Analysis Results</h3>`);

    if (analysisType === 'group_comparison') {
        // Display group comparison results
        displayGroupComparison(container, results);
    } else if (analysisType === 'correlation') {
        // Display correlation results
        displayCorrelationAnalysis(container, results);
    } else if (analysisType === 'longitudinal') {
        // Display longitudinal analysis results
        displayLongitudinalAnalysis(container, results);
    } else if (analysisType === 'classification') {
        // Display classification results
        displayClassificationResults(container, results);
    }
}
function displayGroupComparison(container, results) {
    // Clear any existing content
    container.empty();

    // Check if we have valid results
    if (!results || !results.comparisons || results.comparisons.length === 0) {
        container.append('<div class="alert alert-warning">No comparison data available.</div>');
        return;
    }

    // Create accordion for different metrics
    const accordion = $('<div class="accordion" id="comparisonAccordion"></div>');
    container.append(accordion);

    // Process each comparison
    results.comparisons.forEach((comparison, index) => {
        // Create accordion item
        const metricName = comparison.metric || `Metric ${index + 1}`;
        const accordionItem = `
            <div class="accordion-item">
                <h2 class="accordion-header" id="heading-${index}">
                    <button class="accordion-button ${index === 0 ? '' : 'collapsed'}" type="button" 
                            data-bs-toggle="collapse" data-bs-target="#collapse-${index}" 
                            aria-expanded="${index === 0 ? 'true' : 'false'}" aria-controls="collapse-${index}">
                        <strong>${metricName}</strong>
                        ${comparison.significant ? 
                            '<span class="badge bg-danger ms-2">Significant</span>' : 
                            '<span class="badge bg-secondary ms-2">Non-significant</span>'}
                        <span class="ms-3 text-muted">p-value: ${comparison.p_value.toFixed(4)}</span>
                    </button>
                </h2>
                <div id="collapse-${index}" class="accordion-collapse collapse ${index === 0 ? 'show' : ''}" 
                     aria-labelledby="heading-${index}" data-bs-parent="#comparisonAccordion">
                    <div class="accordion-body">
                        <div id="chart-container-${index}" class="chart-container" style="height: 400px;"></div>
                        <div class="mt-3 table-responsive">
                            <table class="table table-sm table-bordered">
                                <thead>
                                    <tr>
                                        <th>Group</th>
                                        <th>Mean</th>
                                        <th>StdDev</th>
                                        <th>Min</th>
                                        <th>Max</th>
                                        <th>Median</th>
                                        <th>N</th>
                                    </tr>
                                </thead>
                                <tbody id="stats-table-${index}">
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        `;

        accordion.append(accordionItem);

        // Add data to the stats table
        const statsTable = $(`#stats-table-${index}`);
        Object.entries(comparison.group_stats).forEach(([group, stats]) => {
            statsTable.append(`
                <tr>
                    <td><strong>${group}</strong></td>
                    <td>${stats.mean.toFixed(2)}</td>
                    <td>${stats.std.toFixed(2)}</td>
                    <td>${stats.min.toFixed(2)}</td>
                    <td>${stats.max.toFixed(2)}</td>
                    <td>${stats.median.toFixed(2)}</td>
                    <td>${stats.count}</td>
                </tr>
            `);
        });

        // Render the chart after the DOM has been updated
        setTimeout(() => {
            createComparisonChart(`chart-container-${index}`, comparison);
        }, 100);
    });
}

function createComparisonChart(containerId, comparison) {
    // Prepare data for the chart
    const groups = Object.keys(comparison.group_stats);
    const means = groups.map(group => comparison.group_stats[group].mean);
    const errors = groups.map(group => comparison.group_stats[group].std);
    const counts = groups.map(group => comparison.group_stats[group].count);

    // Different colors for each group
    const colors = ['#4e73df', '#1cc88a', '#f6c23e', '#e74a3b', '#36b9cc', '#6f42c1'];

    // Create chart
    const ctx = document.getElementById(containerId);
    const chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: groups,
            datasets: [{
                label: comparison.metric,
                data: means,
                backgroundColor: groups.map((_, i) => colors[i % colors.length]),
                borderColor: groups.map((_, i) => colors[i % colors.length]),
                borderWidth: 1,
                errorBars: {
                    show: true,
                    color: 'black',
                    lineWidth: 2,
                    tipWidth: 6
                }
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `${comparison.metric} Comparison (p=${comparison.p_value.toFixed(4)})`,
                    font: {
                        size: 16
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const index = context.dataIndex;
                            return [
                                `Mean: ${means[index].toFixed(2)}`,
                                `StdDev: ${errors[index].toFixed(2)}`,
                                `N: ${counts[index]}`
                            ];
                        }
                    }
                },
                legend: {
                    display: false
                },
                datalabels: {
                    display: true,
                    align: 'end',
                    anchor: 'end',
                    formatter: function(value, context) {
                        return value.toFixed(1);
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: comparison.metric_units || comparison.metric
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Groups'
                    }
                }
            }
        }
    });

    // Add error bars
    if (Chart.plugins.getAll().findIndex(p => p.id === 'chartjs-plugin-error-bars') !== -1) {
        chart.data.datasets[0].errorBars = {
            yMin: means.map((mean, i) => mean - errors[i]),
            yMax: means.map((mean, i) => mean + errors[i])
        };
        chart.update();
    }
}

function displayCorrelationAnalysis(container, results) {
    // Clear any existing content
    container.empty();

    // Check if we have valid results
    if (!results || !results.correlations || results.correlations.length === 0) {
        container.append('<div class="alert alert-warning">No correlation data available.</div>');
        return;
    }

    // Create table for correlations
    const table = $(`
        <div class="table-responsive">
            <table class="table table-striped table-hover">
                <thead>
                    <tr>
                        <th>Variable X</th>
                        <th>Variable Y</th>
                        <th>Correlation (r)</th>
                        <th>p-value</th>
                        <th>Significance</th>
                        <th>Plot</th>
                    </tr>
                </thead>
                <tbody id="correlation-table-body">
                </tbody>
            </table>
        </div>
    `);

    container.append(table);

    // Container for scatter plots
    const plotContainer = $('<div id="correlation-plots" class="mt-4"></div>');
    container.append(plotContainer);

    // Add rows to the table
    const tableBody = $('#correlation-table-body');
    results.correlations.forEach((corr, index) => {
        const row = $(`
            <tr class="${corr.significant ? 'table-success' : ''}">
                <td>${corr.variable_x}</td>
                <td>${corr.variable_y}</td>
                <td>${corr.r.toFixed(3)}</td>
                <td>${corr.p_value.toFixed(4)}</td>
                <td>${corr.significant ? 
                    '<span class="badge bg-success">Significant</span>' : 
                    '<span class="badge bg-secondary">Non-significant</span>'}</td>
                <td>
                    <button class="btn btn-sm btn-primary show-plot-btn" data-index="${index}">
                        <i class="fas fa-chart-scatter"></i> Plot
                    </button>
                </td>
            </tr>
        `);
        tableBody.append(row);
    });

    // Handle plot button clicks
    $('.show-plot-btn').on('click', function() {
        const index = $(this).data('index');
        const corr = results.correlations[index];

        // Clear existing plots and create new one
        plotContainer.empty();

        const chartContainer = $(`
            <div class="card mb-4">
                <div class="card-header">
                    <h5>Correlation: ${corr.variable_x} vs ${corr.variable_y}</h5>
                    <p class="mb-0">r = ${corr.r.toFixed(3)}, p = ${corr.p_value.toFixed(4)}</p>
                </div>
                <div class="card-body">
                    <canvas id="scatter-plot-${index}" width="400" height="300"></canvas>
                </div>
            </div>
        `);

        plotContainer.append(chartContainer);

        // Create scatter plot
        setTimeout(() => {
            createScatterPlot(`scatter-plot-${index}`, corr);
        }, 100);
    });
}