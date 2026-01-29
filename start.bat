@echo off
title Qwen3-TTS Server
echo Starting Qwen3-TTS Server...
call C:\Users\travi\miniconda3\condabin\conda.bat activate qwen3-tts
cd /d D:\Projects\qwen3-tts-server
python -m api.main
pause
