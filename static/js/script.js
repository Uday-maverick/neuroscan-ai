// Main JavaScript for Alzheimer MRI App
// Handles UI interactions, API calls, and visualizations

document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Initialize tooltips
    initializeTooltips();
    
    // Add smooth scroll behavior
    addSmoothScroll();
    
    // Initialize form handlers
    initializeFormHandlers();
}

/**
 * Initialize Bootstrap tooltips
 */
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Add smooth scroll behavior
 */
function addSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

/**
 * Initialize form handlers
 */
function initializeFormHandlers() {
    // File input handling
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', handleFileInput);
    });
    
    // Upload form submission
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleUploadSubmit);
    }
    
    // Initialize drag and drop
    initializeUploadDragDrop('.upload-area', 'input[type="file"]');
}

/**
 * Handle file input
 */
function handleFileInput(e) {
    const file = e.target.files[0];
    if (file) {
        // Validate file type
        const validTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff'];
        if (!validTypes.includes(file.type)) {
            showNotification('Invalid file type. Please upload JPG, PNG, BMP, or TIFF', 'error');
            return;
        }

        // Validate file size
        const maxSize = 16 * 1024 * 1024; // 16MB
        if (file.size > maxSize) {
            showNotification('File size exceeds 16MB limit', 'error');
            return;
        }

        // Display file info
        const fileInfo = document.getElementById('fileInfo');
        if (fileInfo) {
            fileInfo.innerHTML = `
                <div class="alert alert-info">
                    <strong>File Selected:</strong> ${file.name} (${formatFileSize(file.size)})
                </div>
            `;
        }
        
        showNotification('File selected successfully', 'success');
    }
}

/**
 * Handle upload form submission
 */
async function handleUploadSubmit(e) {
    e.preventDefault();
    
    const fileInput = document.querySelector('input[type="file"]');
    const file = fileInput.files[0];
    
    if (!file) {
        showNotification('Please select a file', 'warning');
        return;
    }
    
    showLoadingSpinner('Analyzing MRI scan...');
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        console.log('Sending file to API:', file.name, file.size, file.type);
        
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        
        console.log('API Response status:', response.status);
        
        const data = await response.json();
        console.log('API Response data:', data);
        
        if (!response.ok || (data && data.success === false)) {
            throw new Error(data?.error || 'Analysis failed');
        }
        
        // Display results
        displayAnalysisResults(data);
        showNotification('Analysis completed successfully!', 'success');
        
    } catch (error) {
        console.error('Analysis error:', error);
        showNotification('Analysis failed: ' + error.message, 'error');
    } finally {
        hideLoadingSpinner();
    }
}

/**
 * Display analysis results
 */
function displayAnalysisResults(data) {
    const resultsSection = document.getElementById('results');
    if (!resultsSection) {
        console.error('Results section not found in DOM');
        return;
    }
    
    // Ensure confidence is a number between 0-1 or 0-100
    let confidence = data.confidence;
    if (confidence > 1) {
        confidence = confidence / 100;
    }
    
    const confidenceColor = getConfidenceColor(confidence);
    const confidencePercent = (confidence * 100).toFixed(1);
    
    resultsSection.innerHTML = `
        <div class="row mt-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header bg-${confidenceColor}">
                        <h5 class="mb-0 text-white">Prediction Result</h5>
                    </div>
                    <div class="card-body">
                        <h4 class="text-${confidenceColor}">${data.prediction || 'Unknown'}</h4>
                        <div class="mb-3">
                            <label class="form-label">Confidence Score</label>
                            <div class="progress">
                                <div class="progress-bar bg-${confidenceColor}" 
                                     style="width: ${confidencePercent}%">
                                    ${confidencePercent}%
                                </div>
                            </div>
                        </div>
                        ${data.risk_level ? `<p class="text-muted"><strong>Risk Level:</strong> ${data.risk_level}</p>` : ''}
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Class Probabilities</h5>
                    </div>
                    <div class="card-body">
                        ${data.probabilities ? Object.entries(data.probabilities).map(([cls, prob]) => {
                            let probValue = prob;
                            if (probValue > 1) {
                                probValue = probValue / 100;
                            }
                            return `
                                <div class="mb-3">
                                    <div class="d-flex justify-content-between mb-1">
                                        <small>${cls}</small>
                                        <small><strong>${(probValue * 100).toFixed(1)}%</strong></small>
                                    </div>
                                    <div class="progress" style="height: 20px;">
                                        <div class="progress-bar" style="width: ${probValue * 100}%"></div>
                                    </div>
                                </div>
                            `;
                        }).join('') : '<p class="text-muted">No probability data available</p>'}
                    </div>
                </div>
            </div>
        </div>
        <div class="row mt-3">
            <div class="col-12">
                <button class="btn btn-primary" onclick="window.print()">Print Results</button>
                <button class="btn btn-success ms-2" onclick="downloadResults('${data.prediction}')">Download Report</button>
                <button class="btn btn-secondary ms-2" onclick="location.reload()">New Analysis</button>
            </div>
        </div>
    `;
}

/**
 * Get confidence color based on confidence level
 */
function getConfidenceColor(confidence) {
    if (confidence >= 0.9) return 'success';
    if (confidence >= 0.7) return 'warning';
    return 'danger';
}

/**
 * Download analysis report
 */
function downloadResults(prediction) {
    const content = `
ALZHEIMER MRI ANALYSIS REPORT
Generated: ${new Date().toLocaleString()}

CLASSIFICATION: ${prediction}

IMPORTANT DISCLAIMER:
This analysis is for educational and research purposes only.
It should not be used as a substitute for professional medical diagnosis.
Always consult with qualified healthcare professionals.
    `;
    
    const element = document.createElement('a');
    element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(content));
    element.setAttribute('download', 'alzheimer_report.txt');
    element.style.display = 'none';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
}

/**
 * Show notification
 */
function showNotification(message, type = 'info') {
    const alertClass = {
        'success': 'alert-success',
        'error': 'alert-danger',
        'warning': 'alert-warning',
        'info': 'alert-info'
    }[type] || 'alert-info';

    const alertHtml = `
        <div class="alert ${alertClass} alert-dismissible fade show" role="alert" style="position: fixed; top: 20px; right: 20px; z-index: 9999; max-width: 400px;">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;

    const alertContainer = document.createElement('div');
    alertContainer.innerHTML = alertHtml;
    document.body.appendChild(alertContainer.firstChild);

    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        const alerts = document.querySelectorAll('.alert');
        if (alerts.length > 0) {
            alerts[alerts.length - 1].remove();
        }
    }, 5000);
}

/**
 * Format file size for display
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

/**
 * Format percentage for display
 */
function formatPercentage(value, decimals = 2) {
    return (value * 100).toFixed(decimals) + '%';
}

/**
 * Fetch API with error handling
 */
async function fetchAPI(url, options = {}) {
    try {
        const response = await fetch(url, options);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('Fetch error:', error);
        throw error;
    }
}

/**
 * Display loading spinner
 */
function showLoadingSpinner(message = 'Processing...') {
    const spinner = document.createElement('div');
    spinner.className = 'loading-overlay active';
    spinner.id = 'loadingSpinner';
    spinner.innerHTML = `
        <div class="text-center">
            <div class="spinner-border mb-3 text-white" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="text-white">${message}</p>
        </div>
    `;
    spinner.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.7);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9998;
    `;
    document.body.appendChild(spinner);
    return spinner;
}

/**
 * Hide loading spinner
 */
function hideLoadingSpinner() {
    const spinner = document.getElementById('loadingSpinner');
    if (spinner) {
        spinner.remove();
    }
}

/**
 * Initialize upload drag and drop
 */
function initializeUploadDragDrop(uploadAreaSelector, fileInputSelector) {
    const uploadArea = document.querySelector(uploadAreaSelector);
    const fileInput = document.querySelector(fileInputSelector);

    if (!uploadArea || !fileInput) return;

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        uploadArea.classList.add('dragover');
    }

    function unhighlight(e) {
        uploadArea.classList.remove('dragover');
    }

    uploadArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        fileInput.files = files;
        
        // Trigger change event
        const event = new Event('change', { bubbles: true });
        fileInput.dispatchEvent(event);
    }
}

/**
 * Debounce function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Throttle function
 */
function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Export functions for global use
window.AlzheimerApp = {
    showNotification,
    showLoadingSpinner,
    hideLoadingSpinner,
    fetchAPI,
    formatFileSize,
    formatPercentage,
    initializeUploadDragDrop,
    debounce,
    throttle,
    downloadResults
};
