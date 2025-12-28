// API Configuration
// This file configures the backend API endpoints for the NeuroScan AI frontend

// Backend API base URL - change this for different environments
const API_BASE_URL = 'https://neuroscan-ai-iobc.onrender.com';

// Export configuration to window object for use in other scripts
window.API_CONFIG = {
    baseUrl: API_BASE_URL,
    endpoints: {
        predict: `${API_BASE_URL}/api/predict`,
        classNames: `${API_BASE_URL}/api/class-names`,
        results: (id) => `${API_BASE_URL}/api/results/${id}`
    }
};

// Log configuration for debugging (can be removed in production)
console.log('API Configuration loaded:', window.API_CONFIG);
