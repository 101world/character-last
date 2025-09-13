@echo off
REM Podman build script for FLUX.1-dev Kohya worker (RunPod compatible)
REM Models will be downloaded at runtime using HF_TOKEN environment variable

echo Building FLUX.1-dev Kohya worker with Podman...
echo Note: Models will be downloaded at runtime on RunPod using HF_TOKEN env var
echo.

REM Build the Podman image (no HF_TOKEN needed at build time)
podman build -t flux-kohya-worker .

if %ERRORLEVEL% EQU 0 (
    echo.
    echo SUCCESS: Podman image built successfully!
    echo.
    echo For RunPod deployment:
    echo 1. Upload this image to your container registry
    echo 2. Set HF_TOKEN environment variable in RunPod
    echo 3. Models will download automatically on first run
    echo.
    echo To test locally:
    echo podman run --device nvidia.com/gpu=all -e HF_TOKEN=your_token -p 8080:8080 flux-kohya-worker
) else (
    echo.
    echo ERROR: Podman build failed!
    echo.
    echo Troubleshooting:
    echo 1. Check your internet connection
    echo 2. Ensure Podman has sufficient resources
    echo 3. Try: podman build --no-cache -t flux-kohya-worker .
    echo.
)
)