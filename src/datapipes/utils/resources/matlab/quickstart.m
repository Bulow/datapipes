% Quickstart - datapipes is a python library that speeds up data
% analysis of timeseries image data such as e.g. LSCI.
% It's faster in two ways: Faster computation, and faster implementation/exploration/iteration

% tl;dr: Time spent waiting for memory transfers >>> Time spent computing. Instead of performing the computations directly, we first record a recipe that computes a single batch/slice. 
% We can then avoid most of the memory transfers and caching of intermediate values that we would otherwise have to do, since each intermediate value is already in vram since it was just computed on the GPU.
% Additionally, by automatically inferring shape logic and storing it with the recipe, the consumers of the output data get fully convolutionally valid data without ever needing to think about input shapes or overlapping batches.

% Start with brrr.m, if you just want to see how fast it is.

%_____________________________________________________
% Tutorial - the basics

% First, let's import datapipes. This also launches the python environment, which takes a few seconds, but is only
% done once per session.
datapipes = MatDatapipes.importDatapipesPython();
ops = datapipes.Ops;
contrast = datapipes.contrast;

%%
% Next, we'll load a dataset

% path = "C:\Workspace\DataAnalysis\compression_paper\input_datasets\brain.rls";
path = "D:\temp_data_safe_to_delete\brain.rls\brain.j2k.h5";

% The matlab wrapper functions all live in the MatDatapipes library
arr = MatDatapipes.loadMatDatapipe(path);

% The type of `arr` is a matlab class called `MatDatapipe`. It pretends to be a
% regular matlab array, except it doesn't actually contain any data until we ask
% for it. Since we didn't need to do anything yet, it is ready for use immediately,
% without waiting for it to load anything up front.

disp(size(arr))
% When we try to use the `MatDatapipe` array, it calls a python function
% under the hood, that loads *just* the data it was asked for and then converts it into matlab's array format.
a = arr(:, :, 1:128);
MatDatapipes.quickPlot(mean(a, 3))
%%

%%
% We are not limited to just *loading* data when asked for it; we can also perform
% computations. In fact, doing it this way is much more efficient than
% saving and moving intermediate calculations back and forth between CPU
% and GPU memory. That's because the relevant input data for the next step in our
% pipeline (i.e. last step's outputs) are already in GPU memory, and most computations on GPUs are MUCH
% faster than moving things from CPU memory (RAM) to GPU memory (VRAM) and vice versa.

% As an example, this pipeline computes the temporal contrast of our
% dataset with a window size of 25:
window_size = int32(25);
temporalContrast = arr.then(ops.bytes_to_float01_gpu).then(contrast.temporal_contrast(window_size));

% The arguments passed to then() are just regular python functions. We'll
% get into specifics later.
% For now, just note that as far as matlab is concerned, `temporalContrast`
% is an array containing the temporal contrast of our entire dataset - even though we haven't done any calculations yet.

tc = MatDatapipes.temporalContrast(:, :, 1:128); % (Now we've computed the first 128 frames.)

% Note that `tc` contains the full 128 (valid) frames we asked for, even though our
% temporal window of 25 frames means we ought to discard `25 // 2 = 12`
% frames from each end to keep all indices convolutionally valid.
disp(size(tc))
MatDatapipes.quickPlot(tc(:, :, 1))
%%
% The MatDatapipe automatically grabbed 128 + 2 * 12 = 152 frames from the
% raw dataset, and computed the 
% When adding functions to a `MatDatapipe`, it automatically probes each
% function and measures its output shape relative to multiple different input shapes. 
% It then solves for an equivalent inverse shape transformation, which it can use to load exactly the right indices to result in the requested output shape.
% If a proper transformation cannot be inferred exactly, you will get an
% error message with instructions showing how to supply one manually.

% If we look at the shapes of each `MatDatapipe`, we can see that
% `temporalContrast` is indeed `2 * 12 = 24` frames shorter than `arr`
disp("size(temporalContrast, 3) = " + size(temporalContrast, 3)); disp("size(arr, 3) = " + size(arr, 3));
%%
MatDatapipes.quickPlot(mean(tc, 3))
%%
% This has important implications for memory consumption.
% Note that the logical size, i.e. the size needed to actually represent our contrast, is much larger than the actual size on disk,
% because our contrast pipeline converted the data to 32-bit floating point
% numbers (a.k.a. single in matlab).
datapipes.utils.benchmarking.print_memory_stats(temporalContrast.getPyHandle())
% This difference can be taken advantage of. We can e.g. compute the temporal mean of all our contrast frames without needing
% to load (or store) the full logical size from RAM:
m = datapipes.sinks.mean(temporalContrast.getPyHandle()); % (Since the function expects a python object, we use getPyHandle() to pass the underlying DataPipe python object of our MatDatapipe)
MatDatapipes.quickPlot(MatDatapipes.tensorToMatArray(m))
%%
m = datapipes.utils.benchmarking.mean_output_benchmark(temporalContrast.getPyHandle())
MatDatapipes.quickPlot(MatDatapipes.tensorToMatArray(m))
%%
% Keeping the raw dataset in a cache in RAM makes it even
% faster.
cachedArr = MatDatapipes.loadMatDatapipeCacheCompressed(path);
%%
cachedArrTemporalContrast = cachedArr...
    .then(ops.bytes_to_float01_gpu)...
    .then(contrast.temporal_contrast(int32(25))...
);
datapipes.utils.benchmarking.print_memory_stats(cachedArrTemporalContrast.getPyHandle())
cachedArrTemporalContrastMean = datapipes.sinks.mean(cachedArrTemporalContrast.getPyHandle());
MatDatapipes.quickPlot(MatDatapipes.tensorToMatArray(cachedArrTemporalContrastMean))
%%
cachedUncompressedArr = MatDatapipes.loadMatDatapipeCachePrefetched(path);
%%
cachedUncompressedArrTemporalContrast = cachedUncompressedArr...
    .then(ops.bytes_to_float01_gpu)...
    .then(contrast.temporal_contrast(int32(25))...
);
datapipes.utils.benchmarking.print_memory_stats(cachedUncompressedArrTemporalContrast.getPyHandle())
cachedUncompressedArrTemporalContrastMean = datapipes.sinks.mean(cachedUncompressedArrTemporalContrast.getPyHandle());
MatDatapipes.quickPlot(MatDatapipes.tensorToMatArray(cachedUncompressedArrTemporalContrastMean))
%%


%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
