function [left, right]=handsSegmentationMask(arr, startFrame, length)
% Computes a segmentation mask from arr(:, :, startFrame:startFrame +
% length)
% arr: MatDatapipe
% startFrame: int (use 0 as default)
% length: int (use 256 as default)
%

datapipes = py.importlib.import_module("datapipes");
hands = py.importlib.import_module("datapipes.analysis.hands");

start_frame = startFrame - 1;

raw = arr.then(datapipes.Ops.bytes_to_float01_gpu);
img_data_tensor = raw.pyGetItem(py.slice(int32(start_frame), int32(length)));

out_tuple = hands.anatomical_segmentation.compute_anatomical_mask(img_data_tensor, use_surface_optimization=py.True); % [both, bboxes, amaps]

maps_dict = MatDatapipes.getItem(out_tuple, int32(2));

left = MatDatapipes.tensorToMatArray(getItem(maps_dict, 'Left'));
right = MatDatapipes.tensorToMatArray(getItem(maps_dict, 'Right'));