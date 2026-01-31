#!/bin/bash
echo "Stopping Qwen3-TTS Server on port 8880..."
pids=$(lsof -ti :8880 2>/dev/null)
if [ -n "$pids" ]; then
    echo "Killing process(es): $pids"
    kill $pids
else
    echo "No process found on port 8880"
fi
echo "Done."
