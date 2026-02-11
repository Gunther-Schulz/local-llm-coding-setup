#!/bin/bash

# Start the tool proxy server
# Activates conda env 'llm' before running

set -e

echo "Starting tool proxy server..."

# Activate conda environment
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate llm

# Navigate to tool-proxy directory
cd "$(dirname "$0")/tool-proxy"

# Run the server with provided args, or use defaults
# Set DEBUG=1 to enable debug logging, or pass --verbose / -v
VERBOSE=
[[ -n "$DEBUG" ]] && VERBOSE="--verbose"

if [ $# -eq 0 ]; then
    echo "No args provided, using defaults:"
    echo "  Port: 8002"
    echo "  Backend URL: http://localhost:8000  (match config/server.env PORT)"
    echo "  Config: config/default_rules.yaml"
    [[ -n "$VERBOSE" ]] && echo "  Debug: enabled (DEBUG=1 or --verbose)"
    python server.py --port 8002 --backend-url http://localhost:8000 --config config/default_rules.yaml $VERBOSE
else
    python server.py "$@"
fi