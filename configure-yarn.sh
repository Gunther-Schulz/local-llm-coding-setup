#!/bin/bash
# Configure YaRN for GLM-4-32B-0414 to extend context from 32K to 64K
# Run this on the RunPod instance

set -e

MODEL_DIR="/workspace/models/glm-4-32b-0414-awq"
CONFIG_FILE="$MODEL_DIR/config.json"

echo "⚙️ Configuring YaRN for extended context (32K → 64K)"
echo "Model directory: $MODEL_DIR"
echo ""

# Check if model exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Model config.json not found at $CONFIG_FILE"
    echo "Please download the model first: bash /workspace/scripts/download-glm4.sh"
    exit 1
fi

# Activate conda environment
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate vllm-test

# Backup original config
if [ ! -f "$CONFIG_FILE.backup" ]; then
    cp "$CONFIG_FILE" "$CONFIG_FILE.backup"
    echo "✅ Backed up original config.json"
fi

# Configure YaRN using Python
python3 << EOF
import json
import sys

config_file = "$CONFIG_FILE"

try:
    # Read config
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Check if YaRN is already configured
    if "rope_scaling" in config:
        rope_scaling = config["rope_scaling"]
        if rope_scaling.get("type") == "yarn" and rope_scaling.get("factor") == 2.0:
            print("✅ YaRN is already configured correctly")
            sys.exit(0)
        else:
            print("⚠️ rope_scaling exists but with different settings, updating...")
    
    # Add YaRN configuration
    config["rope_scaling"] = {
        "type": "yarn",
        "factor": 2.0,
        "original_max_position_embeddings": 32768
    }
    
    # Write config
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ YaRN configuration added successfully")
    print("   - Type: yarn")
    print("   - Factor: 2.0 (extends 32K → 64K)")
    print("   - Original max position: 32768")
    
except Exception as e:
    print(f"❌ Error configuring YaRN: {e}")
    sys.exit(1)
EOF

echo ""
echo "✅ YaRN configuration complete!"
echo "The model can now use up to 64K context tokens."


