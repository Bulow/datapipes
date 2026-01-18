
function datapipe=loadMatDatapipe(path)

MatDatapipes.loadPython();

ds = py.datapipes.datasets.load_dataset(path);
dp = py.datapipes.DataPipe(ds);

datapipe = MatDatapipes.MatDatapipe(dp);




