
function datapipes=importDatapipesPython()

MatDatapipes.loadPython();

py.importlib.import_module("datapipes.utils.set_running_under_matlab");
datapipes = py.importlib.import_module("datapipes");
datapipes.utils.print_gpu_info()
