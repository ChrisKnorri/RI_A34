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

# Wait 5 seconds for initialization
sleep 5

# Start the third command
if command -v gnome-terminal &> /dev/null; then
    # If gnome-terminal is available
    gnome-terminal --tab -- bash -c "python3 Run_Utils.py; exec bash" &
    echo "Started Run_Utils.py in a new gnome-terminal tab with PID $!"
elif command -v xterm &> /dev/null; then
    # If xterm is available
    xterm -hold -e "python3 Run_Utils.py" &
    echo "Started Run_Utils.py in a new xterm window with PID $!"
else
    # If no GUI terminal is available, run in the background
    python3 Run_Utils.py &
    echo "Started Run_Utils.py in the background with PID $!"
fi

# Wait indefinitely to keep the parent process alive
echo "Parent process running. Press Ctrl+C to terminate."
wait
