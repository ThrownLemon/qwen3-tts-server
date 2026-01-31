#!/bin/bash
echo "Starting Qwen3-TTS Server..."
eval "$(conda shell.bash hook)"
conda activate qwen3-tts
export TTS_DEBUG=true
python -m api.main
