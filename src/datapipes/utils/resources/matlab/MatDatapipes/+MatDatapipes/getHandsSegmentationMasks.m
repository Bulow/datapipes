function masks=getHandsSegmentationMasks(path)

hands = py.importlib.import_module("datapipes.analysis.hands");
masks_py = hands.compute_segmentation_masks(path);
masks = MatDatapipes.tensorToMatArray(masks_py);
