// Build script to inject API URL from environment variable
const fs = require('fs');
const path = require('path');

// Get API URL from environment variable (for Render) or use default
const API_URL = process.env.API_URL || 
                process.env.REACT_APP_API_URL || 
                process.env.VITE_API_URL || 
                'https://neuroscan-api.onrender.com';

console.log('🔧 Building frontend with API URL:', API_URL);

// Ensure js directory exists
const jsDir = path.join(__dirname, 'js');
if (!fs.existsSync(jsDir)) {
    console.log('📁 Creating js directory...');
    fs.mkdirSync(jsDir, { recursive: true });
}

// Read config.js
const configPath = path.join(jsDir, 'config.js');

// Check if config.js exists, if not create a default one
if (!fs.existsSync(configPath)) {
    console.log('⚠️  Config file not found, creating default...');
    const defaultConfig = `// API Configuration
// This will be replaced at build time with the actual API URL from environment variable
const API_BASE_URL = '${API_URL}';

// Export for use in other scripts
window.API_CONFIG = {
    baseUrl: API_BASE_URL,
    endpoints: {
        predict: API_BASE_URL + '/api/predict',
        health: API_BASE_URL + '/api/health',
        statistics: API_BASE_URL + '/api/statistics',
        classNames: API_BASE_URL + '/api/class-names',
        results: function(id) { return API_BASE_URL + '/api/results/' + id; }
    }
};
`;
    fs.writeFileSync(configPath, defaultConfig);
    console.log('✅ Created config.js with API URL:', API_URL);
} else {
    // Read and update existing config
    let configContent = fs.readFileSync(configPath, 'utf8');
    
    // Replace the API_BASE_URL with the actual URL from environment
    configContent = configContent.replace(
        /const API_BASE_URL = .*?;/,
        `const API_BASE_URL = '${API_URL}';`
    );
    
    // Write back
    fs.writeFileSync(configPath, configContent, 'utf8');
    console.log('✅ Updated config.js with API URL:', API_URL);
}

console.log('✨ Build completed successfully!');
