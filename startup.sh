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
                       maps-backend.googleapis.com \
                       iam.googleapis.com

# 3. Create Service Account and Credentials
echo "Creating Service Account and Credentials..."
SA_NAME="ee-agent-sa"
SA_EMAIL="$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"

# Create SA if it doesn't exist
if ! gcloud iam service-accounts describe $SA_EMAIL --project=$PROJECT_ID &>/dev/null; then
    gcloud iam service-accounts create $SA_NAME --display-name="Earth Engine Agent SA" --project=$PROJECT_ID
fi

# Grant roles
echo "Granting roles to Service Account..."
# Grant Vertex AI User role, which includes aiplatform.endpoints.predict
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/aiplatform.user" || echo "Warning: Failed to grant Vertex AI role."

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/earthengine.viewer" || echo "Warning: Failed to grant Earth Engine role."

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/serviceusage.serviceUsageConsumer" || echo "Warning: Failed to grant Service Usage Consumer role."

echo "Waiting for IAM role propagation (60 seconds)..."
sleep 5

# Generate key
KEY_FILE="sa-key.json"
echo "Generating Service Account key..."
gcloud iam service-accounts keys create $KEY_FILE \
    --iam-account=$SA_EMAIL --project=$PROJECT_ID

# 4. Setup Local Environment
echo "Setting up dependencies..."
if command -v uv &> /dev/null; then
    uv sync
else
    echo "uv not found, falling back to pip..."
    python3 -m venv .venv
    ./.venv/bin/pip install -e .
fi

# 5. Generate API Key for Maps
echo "Generating Google Maps API Key..."
EXISTING_KEY=$(gcloud services api-keys list --filter="displayName='Earth Engine Agent Key'" --format="value(name)" 2>/dev/null || true)

if [ -z "$EXISTING_KEY" ]; then
    echo "Attempting to create API key..."
    if ! gcloud services api-keys create --display-name="Earth Engine Agent Key" \
        --api-target=service=geocoding-backend.googleapis.com \
        --api-target=service=maps-backend.googleapis.com; then
        echo "Error: Failed to create API key. This may be due to authentication issues."
        echo "Please run the following command manually in Cloud Shell and then restart this script:"
        echo "  gcloud services api-keys create --display-name=\"Earth Engine Agent Key\" --api-target=service=geocoding-backend.googleapis.com --api-target=service=maps-backend.googleapis.com"
        exit 1
    fi
fi

# Get the key string
echo "Retrieving API key string..."
KEY_ID=$(gcloud services api-keys list --filter="displayName='Earth Engine Agent Key'" --format="value(name)" | head -n 1)
KEY_STRING=$(gcloud alpha services api-keys get-key-string $KEY_ID | head -n 1)

if [ -z "$KEY_STRING" ] || [ "$KEY_STRING" = "YOUR_API_KEY" ]; then
    echo "Error: Could not retrieve the API key string."
    echo "Please run 'gcloud services api-keys list' manually and add the key string to .env as GOOGLE_MAPS_API_KEY."
    exit 1
fi

# 6. Configure .env
echo "Configuring .env file..."
cp .env.example .env

# Update variables in .env
sed -i "s|GOOGLE_CLOUD_PROJECT=.*|GOOGLE_CLOUD_PROJECT=\"$PROJECT_ID\"|" .env
sed -i "s|GOOGLE_MAPS_API_KEY=.*|GOOGLE_MAPS_API_KEY=\"$KEY_STRING\"|" .env
sed -i "s|GEEVIZ_MCP_URL=.*|GEEVIZ_MCP_URL=\"https://9001-cs-[PROJECT_HASH].cloudshell.dev/mcp\"|" .env

# Append credentials path
echo "GOOGLE_APPLICATION_CREDENTIALS=\"$KEY_FILE\"" >> .env

# 7. Start Servers
echo "Starting geeViz MCP server in the background..."
export GOOGLE_CLOUD_PROJECT="$PROJECT_ID"
export GOOGLE_MAPS_API_KEY="$KEY_STRING"
export GOOGLE_APPLICATION_CREDENTIALS="$KEY_FILE"

if [ -d ".venv" ]; then
    .venv/bin/python3 run_mcp_server.py &
else
    uv run python3 run_mcp_server.py &
fi

# Wait a bit for the MCP server to start
sleep 5

echo "Starting ADK web server..."
echo "Once started, click the Web Preview button in Cloud Shell to access the agent."
echo "IMPORTANT: You must manually replace [PROJECT_HASH] in your .env file with your actual Cloud Shell Web Preview hash for geeViz to work correctly."

if [ -d ".venv" ]; then
    .venv/bin/adk web --allow_origins 'regex:https://.*.cloudshell.dev'
else
    uv run adk web --allow_origins 'regex:https://.*.cloudshell.dev'
fi
