function dp=loadContrastHandsRLS(rawMatDataPipe, windowSize)

window_size = int32(windowSize);
datapipes = py.importlib.import_module("datapipes");
ops = datapipes.Ops;
contrast = datapipes.contrast;

spatialContrast = rawMatDataPipe.then(ops.bytes_to_float01_gpu).then(contrast.spatial_contrast(window_size));

dp = spatialContrast;