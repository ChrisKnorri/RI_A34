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

# Wait 3 seconds for initialization
sleep 3

# # Start the third command with additional setup
# if command -v gnome-terminal &> /dev/null; then
#     # If gnome-terminal is available
#     gnome-terminal --tab -- bash -c "unset PYTHONPATH && (python3 Run_Utils.py <<< '20'); exec bash" &
#     echo "Started Run_Utils.py in a new gnome-terminal tab with setup commands, PID $!"
# elif command -v xterm &> /dev/null; then
#     # If xterm is available
#     xterm -hold -e "bash -c 'unset PYTHONPATH && (sleep 3; echo 20 | python3 Run_Utils.py)'" &
#     echo "Started Run_Utils.py in a new xterm window with setup commands, PID $!"
# else
#     # If no GUI terminal is available, run in the background
#     (unset PYTHONPATH && (python3 Run_Utils.py <<< '20')) &
#     echo "Started Run_Utils.py in the background with setup commands, PID $!"
# fi

# Start the third command with additional setup
if command -v gnome-terminal &> /dev/null; then
    # If gnome-terminal is available
    gnome-terminal --tab -- bash -c "unset PYTHONPATH && python3 Run_Utils.py; exec bash" &
    echo "Started Run_Utils.py in a new gnome-terminal tab with setup commands, PID $!"
elif command -v xterm &> /dev/null; then
    # If xterm is available
    xterm -hold -e "bash -c 'unset PYTHONPATH && python3 Run_Utils.py" &
    echo "Started Run_Utils.py in a new xterm window with setup commands, PID $!"
else
    # If no GUI terminal is available, run in the background
    (unset PYTHONPATH && python3 Run_Utils.py) &
    echo "Started Run_Utils.py in the background with setup commands, PID $!"
fi

# Wait indefinitely to keep the parent process alive
echo "Parent process running. Press Ctrl+C to terminate."
wait
