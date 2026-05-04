#!/bin/bash

echo "Stopping running ADK web server..."

# Kill ADK web server
PID_ADK=$(pgrep -f "adk web")
if [ -n "$PID_ADK" ]; then
    echo "Killing ADK web server (PID: $PID_ADK)..."
    kill $PID_ADK || true
fi

# Wait a moment for ports to clear
sleep 2

echo "Starting ADK web server..."

if [ -d ".venv" ]; then
    .venv/bin/adk web --allow_origins 'regex:https://.*.cloudshell.dev'
else
    uv run adk web --allow_origins 'regex:https://.*.cloudshell.dev'
fi
