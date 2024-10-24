#!/usr/bin/env bash

# Script to generate & Upload a video to YT Shorts

# Normalize line endings
sed -i.bak 's/\r$//' "$0" && rm "$0.bak"

# Check which interpreter to use (python)
if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "Python not found. Please install Python 3."
  exit 1
fi

# Read .mp/youtube.json file, loop through accounts array, get each id and print every id
youtube_ids=$($PYTHON -c "import json; print('\n'.join([account['id'] for account in json.load(open('.mp/youtube.json'))['accounts']]))")

echo "What account do you want to upload the video to?"

# Print the ids
echo "$youtube_ids"

# Ask for the id
read -p "Enter the id: " id

# Check if the id is in the list
if echo "$youtube_ids" | grep -q "^$id$"; then
  echo "ID found"
else
  echo "ID not found"
  exit 1
fi

# Run python script
$PYTHON src/cron.py youtube "$id"