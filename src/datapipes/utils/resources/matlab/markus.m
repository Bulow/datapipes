datapipes = importDatapipesPython();
%%
path = "path/to/hands.rls";
arr = MatDatapipes.loadHandsRLS(path, false);
MatDatapipes.quickPlot(arr(:, :, 1))
%%
sc = MatDatapipes.loadContrastHandsRLS(arr, 5);
MatDatapipes.quickPlot(sc(:, :, 1))
%%
% Current segments are placeholders to show that we can construct any number of regions (each defined by a line segment) as linear combinations of the vectors we can construct from the coordinates of each joint in the hand.
% Each pixel belongs to the line segment that it has the lowest distance to,
% where the distance metric incorporates connectedness and skin surface topology in addition to euclidean distance.
% There is plenty of room for optimization, as the current version uses some heavy median filters instead of a more optimized graph connectedness approach.
[left, right] = MatDatapipes.handsSegmentationMask(arr, 1, 256);
MatDatapipes.quickPlot(left);
MatDatapipes.quickPlot(right);
%%
timestamps = arr.getTimestamps()

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
