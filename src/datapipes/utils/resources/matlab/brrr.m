% datapipes is a (very fast) lazy-evaluated modular pipeline library, where pipelines (= "datapipes") pretend to be
% a tensor/array while taking care of data loading, batching, memory
% management, multi-threading, and so on automatically under the hood.

% This demo shows prefetching a dataset to RAM in the background, followed
% by computing global_temporal_mean(local_temporal_contrast(all_raw_lsci_frames))
% Please supply a dataset path below (preferably on a local ssd)
% Supported dataset file types: .rls, .j2k.h5, .hdf5, .mp4

datapipes = MatDatapipes.importDatapipesPython();
ops = datapipes.Ops;
contrast = datapipes.contrast;
path = "path/to/a/dataset.rls";
arr = MatDatapipes.loadMatDatapipeCachePrefetched(path);
window_size = int32(25);
temporalContrast = arr.then(ops.bytes_to_float01_gpu).then(contrast.temporal_contrast(window_size));

datapipes.utils.benchmarking.print_memory_stats(temporalContrast.getPyHandle());
% Dataset is actually ready to use already, but we'll block until it is
% fully loaded  and ready for the benchmark so it doesn't get slowed down by IO operations. 
ds = py.getattr(arr.getPyHandle(), "_dataset");     % ignore syntax
ds.block_until_fully_cached();                      % If you comment out this line, you can run the next matlab cell while the dataset is cached in the background.

%%
% Run speed test
m = datapipes.utils.benchmarking.mean_output_benchmark(temporalContrast.getPyHandle());
MatDatapipes.quickPlot(MatDatapipes.tensorToMatArray(m))

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
