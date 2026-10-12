@echo off
rem setup.cmd            -- check what is installed, change nothing
rem setup.cmd --install  -- install dependencies and download models (~2.8 GB)
rem Comments are ASCII on purpose: cmd reads .cmd in the OEM codepage,
rem and Cyrillic in a UTF-8 file breaks parsing.
setlocal
set "PYTHONPATH=%~dp0src"
set "PYTHONIOENCODING=utf-8"
set "LAWS_ROOT=%~dp0."
"%~dp0.venv\Scripts\python.exe" -m laws_mcp.setup_check %*
