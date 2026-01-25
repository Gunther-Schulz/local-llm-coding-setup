#!/bin/bash
# Configuration Manager Library
# Handles reading/writing centralized .llm-config file

CONFIG_FILE="${ROOT:-$(pwd)}/.llm-config"

# Initialize config file if it doesn't exist
init_config() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        cat > "$CONFIG_FILE" <<'EOF'
# LLM Configuration
# This file is managed by ./select-model.sh
# Do not edit manually unless you know what you're doing

[model]
key=
selected_at=

[context]
mode=normal
# Mode options: normal, extended
# normal = base context (e.g., 32K), fast, GPU only
# extended = extended context (e.g., 128K), slower, GPU+CPU offload

[runtime]
# Override context mode temporarily by setting env var:
# EXTENDED_CONTEXT_MODE=1 ./start-all-vllm.sh
EOF
    fi
}

# Read a config value
# Usage: read_config "section" "key"
read_config() {
    local section="$1"
    local key="$2"
    
    if [[ ! -f "$CONFIG_FILE" ]]; then
        return 1
    fi
    
    # Parse INI-style config
    awk -F= -v section="[$section]" -v key="$key" '
        $0 == section { in_section=1; next }
        /^\[/ { in_section=0 }
        in_section && $1 == key { print $2; exit }
    ' "$CONFIG_FILE" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
}

# Write a config value
# Usage: write_config "section" "key" "value"
write_config() {
    local section="$1"
    local key="$2"
    local value="$3"
    
    init_config
    
    # Create temp file
    local tmp_file="${CONFIG_FILE}.tmp"
    
    # Update or add the key
    awk -F= -v section="[$section]" -v key="$key" -v value="$value" '
        BEGIN { in_section=0; found=0 }
        $0 == section { in_section=1; print; next }
        /^\[/ && in_section { 
            if (!found) print key "=" value
            in_section=0
            found=1
        }
        in_section && $1 == key { 
            print key "=" value
            found=1
            next
        }
        { print }
        END { if (in_section && !found) print key "=" value }
    ' "$CONFIG_FILE" > "$tmp_file"
    
    mv "$tmp_file" "$CONFIG_FILE"
}

# Get current model key
get_current_model() {
    read_config "model" "key"
}

# Set current model
set_current_model() {
    local model_key="$1"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    write_config "model" "key" "$model_key"
    write_config "model" "selected_at" "$timestamp"
}

# Get context mode (returns: normal or extended)
get_context_mode() {
    local mode=$(read_config "context" "mode")
    echo "${mode:-normal}"
}

# Set context mode
set_context_mode() {
    local mode="$1"
    
    if [[ "$mode" != "normal" && "$mode" != "extended" ]]; then
        echo "Error: Invalid context mode: $mode" >&2
        echo "Valid modes: normal, extended" >&2
        return 1
    fi
    
    write_config "context" "mode" "$mode"
}

# Get extended context mode as 0/1 for compatibility
get_extended_context_mode() {
    local mode=$(get_context_mode)
    if [[ "$mode" == "extended" ]]; then
        echo "1"
    else
        echo "0"
    fi
}

# Set extended context mode from 0/1 for compatibility
set_extended_context_mode() {
    local value="$1"
    
    if [[ "$value" == "1" ]]; then
        set_context_mode "extended"
    else
        set_context_mode "normal"
    fi
}

# Show current configuration
show_config() {
    init_config
    
    local model_key=$(get_current_model)
    local mode=$(get_context_mode)
    local selected_at=$(read_config "model" "selected_at")
    
    echo "════════════════════════════════════════════════════════════════"
    echo "  Current Configuration"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    
    if [[ -n "$model_key" ]]; then
        echo "Model:        $model_key"
        if [[ -n "$selected_at" ]]; then
            echo "Selected:     $selected_at"
        fi
    else
        echo "Model:        (none selected)"
    fi
    
    echo ""
    
    if [[ "$mode" == "extended" ]]; then
        echo "Context Mode: 🟡 Extended (128K, slower)"
    else
        echo "Context Mode: 🟢 Normal (32K, fast)"
    fi
    
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "Config file: $CONFIG_FILE"
    echo ""
}

# Validate configuration
validate_config() {
    local model_key=$(get_current_model)
    local mode=$(get_context_mode)
    
    if [[ -z "$model_key" ]]; then
        echo "Error: No model selected" >&2
        return 1
    fi
    
    if [[ "$mode" != "normal" && "$mode" != "extended" ]]; then
        echo "Error: Invalid context mode: $mode" >&2
        return 1
    fi
    
    return 0
}

# Migrate from old dot-files
migrate_from_dotfiles() {
    local migrated=0
    
    # Migrate .current-model
    if [[ -f "$ROOT/.current-model" ]]; then
        local model_key=$(cat "$ROOT/.current-model" | head -1)
        if [[ -n "$model_key" ]]; then
            set_current_model "$model_key"
            echo "✓ Migrated model selection from .current-model" >&2
            migrated=1
        fi
        mv "$ROOT/.current-model" "$ROOT/.current-model.bak"
    fi
    
    # Migrate .context-mode
    if [[ -f "$ROOT/.context-mode" ]]; then
        local mode_value=$(cat "$ROOT/.context-mode" | head -1)
        set_extended_context_mode "$mode_value"
        echo "✓ Migrated context mode from .context-mode" >&2
        migrated=1
        mv "$ROOT/.context-mode" "$ROOT/.context-mode.bak"
    fi
    
    if [[ $migrated -eq 1 ]]; then
        echo "" >&2
        echo "✓ Migration complete! Old files backed up with .bak extension" >&2
        echo "" >&2
    fi
    
    return $migrated
}
