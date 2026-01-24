#!/bin/bash
# Show current configuration

ROOT="${WORKSPACE:-${RUNPOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}}"
source "$ROOT/lib/config-manager.sh"

# Auto-migrate from old files
migrate_from_dotfiles 2>/dev/null

show_config
