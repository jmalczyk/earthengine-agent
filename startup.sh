#!/bin/bash

# Exit on error
set -e

echo "Starting setup for Earth Engine Geospatial Agent in Cloud Shell..."

# 1. Get Project ID
PROJECT_ID=$(gcloud config get-value project)
echo "Using Project ID: $PROJECT_ID"

# 2. Enable Necessary APIs
echo "Enabling required APIs..."
gcloud services enable earthengine.googleapis.com \
                       aiplatform.googleapis.com \
                       geocoding-backend.googleapis.com \
                       maps-backend.googleapis.com

# 3. Setup Local Environment
echo "Setting up dependencies..."
if command -v uv &> /dev/null; then
    uv sync
else
    echo "uv not found, falling back to pip..."
    python3 -m venv .venv
    ./.venv/bin/pip install -e .
fi

# 4. Generate API Key
echo "Generating Google Maps API Key..."
EXISTING_KEY=$(gcloud services api-keys list --filter="displayName='Earth Engine Agent Key'" --format="value(name)" 2>/dev/null || true)

if [ -z "$EXISTING_KEY" ]; then
    gcloud services api-keys create --display-name="Earth Engine Agent Key" \
        --api-target=service=geocoding-backend.googleapis.com \
        --api-target=service=maps-backend.googleapis.com
fi

# Get the key string
KEY_STRING=$(gcloud services api-keys list --filter="displayName='Earth Engine Agent Key'" --format="value(keyString)" | head -n 1)

if [ -z "$KEY_STRING" ]; then
    echo "Warning: Could not automatically retrieve the API key string."
    KEY_STRING="YOUR_API_KEY"
fi

# 5. Configure .env
echo "Configuring .env file..."
cp .env.example .env

# Update variables in .env
sed -i "s|GOOGLE_CLOUD_PROJECT=.*|GOOGLE_CLOUD_PROJECT=\"$PROJECT_ID\"|" .env
sed -i "s|GOOGLE_MAPS_API_KEY=.*|GOOGLE_MAPS_API_KEY=\"$KEY_STRING\"|" .env

# 6. Start Servers
echo "Starting geeViz MCP server in the background..."
export GOOGLE_CLOUD_PROJECT="$PROJECT_ID"
export GOOGLE_MAPS_API_KEY="$KEY_STRING"

if [ -d ".venv" ]; then
    .venv/bin/python3 run_mcp_server.py &
else
    uv run python3 run_mcp_server.py &
fi

# Wait a bit for the MCP server to start
sleep 5

echo "Starting ADK web server..."
echo "Once started, click the Web Preview button in Cloud Shell to access the agent."

if [ -d ".venv" ]; then
    .venv/bin/adk web --allow_origins 'regex:https://.*.cloudshell.dev'
else
    uv run adk web --allow_origins 'regex:https://.*.cloudshell.dev'
fi
