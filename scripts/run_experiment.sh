#!/usr/bin/env bash

# Path to your Python script
python_script="scripts/synthetic_evaluation.py"

# Function to run the Python script
run_script() {
  # Determine if we are on macOS to use caffeinate, otherwise ignore
  local cmd_prefix=""
  if [[ "$OSTYPE" == "darwin"* ]]; then
    # Only use caffeinate if available on macOS
    if command -v caffeinate >/dev/null 2>&1; then
      cmd_prefix="caffeinate"
    fi
  fi

  # Run the Python script
  $cmd_prefix python -u "$python_script" 2>&1 | {
    python_pid=$!
    completed=false

    while IFS= read -r line; do
      echo "$line"
      
      # Check for UserWarning
      if echo "$line" | grep -q "UserWarning: resource_tracker: There appear to be"; then
        echo "Warning detected: Killing process..."
        # Kill the group so we don't leave orphaned children
        kill -9 "$python_pid" 2>/dev/null
        return 1
      fi

      # Check for successful completion
      if echo "$line" | grep -q "Benchmark completed."; then
        echo "Experiment completed successfully."
        completed=true
      fi
    done

    [ "$completed" = true ]
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