@echo off
set NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe
set CCBIN=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64
"%NVCC%" -shared -o "nvfp4_rope_silu.dll" "nvfp4_rope_silu.cu" -O3 -arch=sm_70 -ccbin "%CCBIN%"
if exist "nvfp4_rope_silu.dll" (echo BUILD_OK) else (echo BUILD_FAIL)
