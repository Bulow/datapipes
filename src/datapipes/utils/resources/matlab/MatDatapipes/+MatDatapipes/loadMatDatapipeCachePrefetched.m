function datapipe=loadMatDatapipeCachePrefetched(path)

MatDatapipes.loadPython();

ds = py.datapipes.datasets.load_dataset(path);
cached_ds = py.datapipes.datasets.modifiers.cached_dataset.CachedDataset(ds);
dp = py.datapipes.DataPipe(cached_ds);

datapipe = MatDatapipes.MatDatapipe(dp);
