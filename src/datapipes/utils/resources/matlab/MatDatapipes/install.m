function install()
root = fileparts(mfilename('fullpath'));

if ~isfolder(root)
    error('MatDatapipes:InstallFailed','Install root not found.');
end

addpath(root);                  % so install/uninstall is found
addpath(fullfile(root, '+MatDatapipes')); % package folder is enough

status = savepath;
if status ~= 0
    error("MatDatapipes:InstallFailed","Failed to save path.");
end
end
