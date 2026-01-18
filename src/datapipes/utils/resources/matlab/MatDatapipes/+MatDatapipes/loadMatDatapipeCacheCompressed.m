function datapipe=loadMatDatapipeCacheCompressed(path)

MatDatapipes.loadPython();

ds = py.datapipes.datasets.load_dataset(path);
cached_ds = py.datapipes.datasets.modifiers.compressed_cached_dataset.CompressedCachedDataset(ds);


dp = py.datapipes.DataPipe(cached_ds);

datapipe = MatDatapipes.MatDatapipe(dp);
