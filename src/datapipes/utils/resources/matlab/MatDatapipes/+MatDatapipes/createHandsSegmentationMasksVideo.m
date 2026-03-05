function masks=createHandsSegmentationMasksVideo(in_path, out_path)

hands = py.importlib.import_module("datapipes.analysis.hands");
masks_py = hands.create_segmentation_masks_video(in_path, out_path);
masks = MatDatapipes.tensorToMatArray(masks_py);
