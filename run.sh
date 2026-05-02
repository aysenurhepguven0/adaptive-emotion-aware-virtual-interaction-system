#!/bin/bash
# Activate the virtual environment and launch the GUI
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Run setup.sh first."
    exit 1
fi

# macOS: suppress tkinter deprecation warning
export TK_SILENCE_DEPRECATION=1

.venv/bin/python gui_app.py "$@"
