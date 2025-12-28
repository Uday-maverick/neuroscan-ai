# NeuroScan AI - Frontend

Frontend for Alzheimer's MRI classification web application.

## Deployment on Render

This service can be deployed as a static site on Render.

### Configuration

- **Environment**: Static Site
- **Build Command**: `npm install && npm run build`
- **Publish Path**: `./`

### Environment Variables

- `REACT_APP_API_URL`: Backend API URL (e.g., `https://your-backend.onrender.com`)
- `VITE_API_URL`: Backend API URL (e.g., `https://your-backend.onrender.com`)
- `API_URL`: Backend API URL (e.g., `https://your-backend.onrender.com`)

## Local Development

```bash
npm install -g http-server
npm run build  # This injects the API URL from environment variables
http-server -p 3000
```

Note: The build process will inject the API URL from environment variables into the config.js file.