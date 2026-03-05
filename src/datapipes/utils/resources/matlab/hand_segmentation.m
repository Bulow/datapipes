datapipes = MatDatapipes.importDatapipesPython(); %[output:35a9a3d6]

path = "path/to/hands.rls";
%%


MatDatapipes.createHandsSegmentationMasksVideo(path, "out_vids_matlab/masks.mp4"); %[output:07ccaee7]
%%

arr = MatDatapipes.getHandsSegmentationMasks(path);
MatDatapipes.quickPlot(arr(:, :, 1))

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
%[output:35a9a3d6]
%   data: {"dataType":"text","outputData":{"text":"PyTorch version: 2.10.0+cu128\nCUDA available: True\nCUDA version: 12.8\nNumber of GPUs: 1\nCurrent GPU: NVIDIA GeForce RTX 5090\n","truncated":false}}
%---
%[output:07ccaee7]
%   data: {"dataType":"text","outputData":{"text":"accumulate [--------------------------------------------------]\naccumulate [|||||||||||||||||||||||||||","truncated":false}}
%---
