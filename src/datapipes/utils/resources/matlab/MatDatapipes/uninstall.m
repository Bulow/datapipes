function uninstall()
%UNINSTALL Remove MyLib from the MATLAB path.
%
% Assumes this file (uninstall.m) lives in the library root.
% Removes the library root (and its subfolders) from the path and saves it.

root = fileparts(mfilename('fullpath'));

% Remove root and everything under it from the MATLAB path
pathsToRemove = genpath(root);
if ~isempty(pathsToRemove)
    rmpath(pathsToRemove);
end

% Also remove root explicitly (genpath sometimes behaves oddly with trailing separators)
rmpath(root);

% Persist the change
status = savepath;
if status ~= 0
    error("MatDatapipes:UninstallFailed", ...
        "Failed to save MATLAB path. Try running MATLAB with permissions to write the pathdef.");
end

fprintf("MatDatapipes uninstalled (removed from path): %s\n", root);
end
