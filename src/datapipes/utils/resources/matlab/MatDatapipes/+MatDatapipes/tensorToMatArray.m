function arr = tensorToMatArray(pyArr)
try

    fortranArrPy = py.datapipes.ops.Ops.py_to_matlab(pyArr);
    
    dtypeNpName  = char(py.str(fortranArrPy.dtype.name)); % e.g., 'float64'

    switch dtypeNpName
        case 'float64', arr = double(fortranArrPy);
        case 'float32', arr = single(fortranArrPy);
        case 'int64',   arr = int64(fortranArrPy);
        case 'int32',   arr = int32(fortranArrPy);
        case 'int16',   arr = int16(fortranArrPy);
        case 'int8',    arr = int8(fortranArrPy);
        case 'uint64',  arr = uint64(fortranArrPy);
        case 'uint32',  arr = uint32(fortranArrPy);
        case 'uint16',  arr = uint16(fortranArrPy);
        case 'uint8',   arr = uint8(fortranArrPy);
        case {'bool', 'bool_'}, arr = logical(fortranArrPy);
        otherwise,      error("Unsupported dtype: " + dtypeNpName);
    end
    

catch ME
    % Fallback: return original Python object if conversion fails.
    warning('Python to matlab data type conversion failed: %s', ME.message);
    ME.stack
    ME.cause
    arr = pyArr;
end

end