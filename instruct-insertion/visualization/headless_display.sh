#!/bin/bash
# Script for setting up a headless display for PyVista in remote servers.
# Usage: source headless_display.sh

export DISPLAY=:99.0
export PYVISTA_OFF_SCREEN=true
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
sleep 3
