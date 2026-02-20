function masks=getHandsSegmentationMasks(path)

datapipes = py.importlib.import_module("datapipes");
masks_py = datapipes.analysis.hands.rls_to_segmentation_mask.compute_segmentation_masks(path);
masks = MatDatapipes.tensorToMatArray(masks_py);
