@echo off
echo Stopping Qwen3-TTS Server on port 8880...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8880 ^| findstr LISTENING') do (
    echo Killing process %%a
    taskkill /F /PID %%a >nul 2>&1
)
echo Done.
