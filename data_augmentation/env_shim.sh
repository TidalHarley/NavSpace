# shellcheck shell=bash
# Source this file BEFORE running any habitat-sim script in this pipeline.
#
#   conda activate navspace39
#   source data_augmentation/env_shim.sh
#
# It forces glvnd to load the NVIDIA EGL/GLX libs instead of the conda env's
# Mesa libGL, which otherwise crashes habitat-sim 0.2.5 with
# "cannot retrieve OpenGL version: InvalidValue".
# Same trick as snav_training/data_generation/run_render_r2rce.sh.

NVIDIA_GLX_LIB="${NVIDIA_GLX_LIB:-/lib/x86_64-linux-gnu/libGLX_nvidia.so.0}"
NVIDIA_GL_DISPATCH_LIB="${NVIDIA_GL_DISPATCH_LIB:-/lib/x86_64-linux-gnu/libGLdispatch.so.0}"
if [ -z "${LD_PRELOAD:-}" ] && [ -e "${NVIDIA_GLX_LIB}" ] && [ -e "${NVIDIA_GL_DISPATCH_LIB}" ]; then
  export LD_PRELOAD="${NVIDIA_GLX_LIB}:${NVIDIA_GL_DISPATCH_LIB}"
fi

NVIDIA_EGL_ICD="${NVIDIA_EGL_ICD:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
if [ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ] && [ -e "${NVIDIA_EGL_ICD}" ]; then
  export __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_EGL_ICD}"
fi

if [ -z "${DASHSCOPE_API_KEY:-}" ]; then
  echo "[env_shim] WARN: DASHSCOPE_API_KEY is not set. Qwen calls will fail." >&2
fi

echo "[env_shim] LD_PRELOAD=${LD_PRELOAD:-<unset>}"
echo "[env_shim] __EGL_VENDOR_LIBRARY_FILENAMES=${__EGL_VENDOR_LIBRARY_FILENAMES:-<unset>}"
