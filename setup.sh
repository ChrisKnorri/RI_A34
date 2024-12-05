#!/bin/bash

# Function to handle termination
cleanup() {
    kill -TERM -- -$$ 2>/dev/null
    exit 0
}

# Trap SIGINT and SIGTERM to clean up
trap cleanup INT TERM

# Start the first command
rcssserver3d &
echo "Started rcssserver3d with PID $!"

# Start the second command
sh RoboViz/bin/roboviz.sh &
echo "Started roboviz.sh with PID $!"

# Wait 10 security seconds
sleep 5

# Start the third command in a new terminal tab
gnome-terminal --tab -- bash -c "python3 Run_Utils.py; exec bash" &
echo "Started Run_Utils.py in a new terminal tab with PID $!"

# Wait indefinitely to keep the parent process alive
echo "Parent process running. Press Ctrl+C to terminate."
wait
