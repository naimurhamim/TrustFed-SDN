@echo off
REM Convenience wrapper on Windows forwarding to MSYS2 mingw32-make
if exist "C:\msys64\mingw64\bin\mingw32-make.exe" (
    "C:\msys64\mingw64\bin\mingw32-make.exe" %*
) else (
    mingw32-make %*
)

