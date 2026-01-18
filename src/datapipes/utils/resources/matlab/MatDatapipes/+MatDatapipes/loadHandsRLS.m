function datapipe=loadHandsRLS(path, block)

MatDatapipes.loadPython();

ds = py.datapipes.datasets.DatasetRLS(path, switch_wh_metadata_read_order=py.True);
cached_ds = py.datapipes.datasets.modifiers.cached_dataset.CachedDataset(ds);



if (block)
    cached_ds.block_until_fully_cached();
end

dp = py.datapipes.DataPipe(cached_ds);

datapipe = MatDatapipes.MatDatapipe(dp);
