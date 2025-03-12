#!/bin/zsh

# Path to your Python script
python_script="scripts/synthetic_evaluation.py"

# Function to run the Python script
run_script() {
  # Start the Python process in the background and capture its PID
  unbuffer caffeinate python $python_script 2>&1 | {
    python_pid=$!  # Capture the PID of the unbuffered Python process
    completed=false  # Flag to detect "All experiments completed."

    while IFS= read -r line; do
      echo "$line"
      
      # Check for UserWarning
      if echo "$line" | grep -q "UserWarning: resource_tracker: There appear to be"; then
        echo "Warning detected: Terminating the script..."
        pkill -P $python_pid  # Safely terminate the Python process
        return 1  # Return non-zero to indicate warning was found
      fi

      # Check for successful completion
      if echo "$line" | grep -q "Benchmark completed."; then
        echo "Experiment completed successfully."
        completed=true  # Set the completion flag
      fi
    done

    # Only return success if "All experiments completed." was found
    if [ "$completed" = true ]; then
      return 0
    else
      echo "Script terminated without completing all experiments."
      return 1
    fi
  }
}

# Loop to continuously run the script until no warning is detected and it completes
while true; do
  echo "Starting the Python script..."
  if run_script; then
    echo "Script completed successfully and without warnings. Exiting."
    break
  else
    echo "Restarting the script..."
    sleep 1
  fi
done