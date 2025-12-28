// Build script to inject API URL from environment variable
const fs = require('fs');
const path = require('path');

// Get API URL from environment variable (for Render) or use default
const API_URL = process.env.REACT_APP_API_URL || 
                process.env.VITE_API_URL || 
                process.env.API_URL || 
                'https://neuroscan-api.onrender.com';

// Read config.js
const configPath = path.join(__dirname, 'js', 'config.js');
let configContent = fs.readFileSync(configPath, 'utf8');

// Replace the API_BASE_URL with the actual URL from environment
configContent = configContent.replace(
    /const API_BASE_URL = .*?;/,
    `const API_BASE_URL = '${API_URL}';`
);

// Write back
fs.writeFileSync(configPath, configContent, 'utf8');

console.log(`✓ Injected API URL: ${API_URL}`);

