@echo off
rem laws.cmd "question" [more questions ...] -- search over Russian codes
rem Comments are ASCII on purpose: cmd reads .bat/.cmd in the OEM codepage,
rem and Cyrillic in a UTF-8 file breaks parsing.
setlocal
set "PYTHONPATH=%~dp0src"
set "PYTHONIOENCODING=utf-8"
set "LAWS_ROOT=%~dp0."
"%~dp0.venv\Scripts\python.exe" -m laws_mcp.cli %*
