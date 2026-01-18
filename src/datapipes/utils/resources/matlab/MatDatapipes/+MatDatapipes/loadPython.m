
function loadPython()

if isempty(pyenv)
    disp("[Datapipes]: Loading python environment - this takes a few seconds and is only done once per session.")
    pyenv(Version=".datapipes-python\.venv\Scripts\python.exe", ExecutionMode="OutOfProcess") %, ExecutionMode="InProcess"); %
end

