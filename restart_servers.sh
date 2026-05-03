#!/bin/bash

echo "Stopping running servers..."

# Kill GeeViz MCP server
PID_MCP=$(pgrep -f "run_mcp_server.py")
if [ -n "$PID_MCP" ]; then
    echo "Killing GeeViz MCP server (PID: $PID_MCP)..."
    kill $PID_MCP || true
fi

# Kill ADK web server
PID_ADK=$(pgrep -f "adk web")
if [ -n "$PID_ADK" ]; then
    echo "Killing ADK web server (PID: $PID_ADK)..."
    kill $PID_ADK || true
fi

# Wait a moment for ports to clear
sleep 2

echo "Starting servers..."

# Start GeeViz MCP in background
if [ -d ".venv" ]; then
    .venv/bin/python3 run_mcp_server.py &
else
    uv run python3 run_mcp_server.py &
fi

# Wait for it to start
sleep 5

echo "Starting ADK web server..."
if [ -d ".venv" ]; then
    .venv/bin/adk web --allow_origins 'regex:https://.*.cloudshell.dev'
else
    uv run adk web --allow_origins 'regex:https://.*.cloudshell.dev'
fi
