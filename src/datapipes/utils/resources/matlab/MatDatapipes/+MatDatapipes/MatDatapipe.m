classdef MatDatapipe %< matlab.mixin.indexing.RedefinesParen
    % DataPipeWrapper
    %
    % A thin MATLAB wrapper around a Python `DataPipe` object that exposes
    % MATLAB-native indexing and size/shape behaviors.
    %
    % WHY THIS EXISTS
    % ---------------
    % - Python uses 0-based indexing and `__getitem__(int|slice)` for access.
    % - MATLAB uses 1-based indexing and `()` with colon `:` and ranges like `i:j`.
    % - This wrapper translates MATLAB-style indexing into the Python methods
    %   that `DataPipe` already implements.
    %
    % REQUIREMENTS
    % ------------
    % - MATLAB must have Python integration enabled by calling `loadPython` when loading datapipes via loadMatDatapipe, this is done automatically.
    % - The Python module that defines `DataPipe` must be on the Python path.
    % - Your `DataPipe` Python class must provide:
    %     * __getitem__(index: int|slice) -> numpy.ndarray   % (NO step support)
    %     * __len__() -> int
    %     * shape property (tuple[int, ...])                  % e.g. (T, H, W, C)
    %
    % WHAT YOU GET
    % ------------
    % - MATLAB-friendly indexing:
    %     dp = DataPipeWrapper(py.some_module.DataPipe(...));
    %     x  = dp(1);           % 1-based -> __getitem__(0)
    %     x  = dp(5:10);        % inclusive range -> __getitem__(slice(4, 10))
    %     x  = dp(:);           % full range -> __getitem__(slice(0, len))
    %
    %   You can also index further into the returned array using MATLAB syntax:
    %     frame = dp(7)(:, :, 1);         % Python ndarray returned then MATLAB-indexed
    %     patch = dp(20:30)(1:32,1:32,:); % slice then MATLAB-indexing on the result
    %
    % LIMITATIONS (by design to match DataPipe)
    % -----------------------------------------
    % - Step values in ranges are NOT supported (e.g., `1:2:10`). If provided,
    %   an error is raised (DataPipe ignores step; we avoid silently returning wrong data).
    % - Non-contiguous index vectors (e.g., [1 3 4 10]) are NOT supported and will error,
    %   unless they are a contiguous run with unit step (which is converted to a slice).
    %
    % RETURN TYPE
    % -----------
    % - The wrapper converts returned NumPy ndarrays into MATLAB numeric arrays
    %   (preserving shape and element order). If the dtype is not supported below,
    %   the original Python object is returned.
    %
    % SUPPORTED NUMPY DTYPES FOR CONVERSION
    % -------------------------------------
    % - float64, float32
    % - int64, int32, int16, int8
    % - uint64, uint32, uint16, uint8
    % - bool
    %
    % NOTE FOR PYTHON USERS NEW TO MATLAB
    % -----------------------------------
    % - MATLAB is 1-based: the first element is at index 1, not 0.
    % - MATLAB ranges are inclusive on the right (e.g., `5:10` includes 10).
    % - MATLAB stores arrays in column-major order (like Fortran), whereas NumPy
    %   uses row-major (C). The converter below accounts for ordering so you get
    %   the expected layout when working in MATLAB.
    %
    % -------------------------------------------------------------------------

    properties (SetAccess = private)
        % The underlying Python DataPipe object
        pyobj
        pyGetItem
        pyToMatlabLayout
    end

    methods
        function obj = MatDatapipe(pyDataPipe)
            % Constructor
            %
            % Usage:
            %   dp = DataPipeWrapper(datapipes.DataPipe(args...))
            %
            % You can either pass an already-constructed Python DataPipe instance:
            %   py_dp = datapipes.DataPipe(args...);
            %   dp    = MatDatapipe(py_dp);
            %
            % Or construct inline as above.
            if nargin ~= 1
                error('MatDatapipe:Constructor', 'Provide a single Python Datapipe instance.');
            end
            % We duck-type: only minimal checks to ensure required members exist.
            if ~py.hasattr(pyDataPipe, '__getitem__')
                error('MatDatapipe:Constructor', 'Given object lacks __getitem__.');
            end
            if ~py.hasattr(pyDataPipe, '__len__')
                error('MatDatapipe:Constructor', 'Given object lacks __len__.');
            end
            if ~py.hasattr(pyDataPipe, 'shape')
                error('MatDatapipe:Constructor', 'Given object lacks shape property.');
            end
            obj.pyobj = pyDataPipe;
            obj.pyGetItem = py.getattr(obj.pyobj, '__getitem__');
        end

        function pyHandle = getPyHandle(obj)
            pyHandle = obj.pyobj;
        end

        function newObj = then(obj, pyFunction)
            newPyDatapipe = obj.pyobj.then(pyFunction);
            newObj = MatDatapipes.MatDatapipe(newPyDatapipe);
        end

        function n = length(obj)
            % MATLAB `length` -> number of elements along the first dimension (time axis).
            n = double(py.len(obj.pyobj));
        end

        function n = numel(obj, varargin)
            % Total number of elements in the array (product of shape dims).
            shp = obj.shape();
            if isempty(shp)
                n = 0;
            else
                n = prod(shp);
            end
        end

        function s = size(obj, dim)
            % MATLAB `size` behavior.
            %
            % - size(obj) returns the shape vector, cast to double.
            % - size(obj, k) returns the k-th dimension size (with trailing 1s padded).
            shp = obj.shape();
            if nargin == 1
                s = shp;
            else
                if dim <= numel(shp)
                    s = shp(dim);
                else
                    % As in MATLAB, requesting size beyond ndims returns 1
                    s = 1;
                end
            end
        end

        function p = getPyDatapipe(obj)
            p = obj.pyobj;
        end

        function d = ndims(obj)
            % Number of dimensions (length of shape).
            d = numel(obj.shape());
        end

        function ind = end(obj, k, n)
            % Support MATLAB `end` keyword in indexing.
            %
            % Example:
            %   dp(1:end)   % full range
            %   dp(5:end)   % from 5 to end
            shp = obj.shape();
            if k == 1
                ind = shp(1); % end refers to the first dimension in dp(...)
            else
                if k <= numel(shp)
                    ind = shp(k);
                else
                    ind = 1;
                end
            end
        end

        function varargout = subsref(obj, S)
            % Core MATLAB indexing bridge.
            %
            % This method maps MATLAB-style () indexing to Python __getitem__.
            % Only the first subscript is forwarded to DataPipe; additional
            % subscripts are applied on the returned ndarray/MATLAB array.
            %
            % Supported forms:
            %   dp(i)        -> __getitem__(i-1)
            %   dp(i:j)      -> __getitem__(slice(i-1, j))   % inclusive right
            %   dp(:)        -> __getitem__(slice(0, len))
            %
            % Disallowed (errors):
            %   dp(1:2:10)      % step != 1
            %   dp([1 3 4 10])  % non-contiguous
            %
            switch S(1).type
                case '()'
                    idx = S(1).subs;

                    if isempty(idx)
                        error('DatapipeWrapper:subsref', 'Indexing requires at least one subscript.');
                    end

                    % Translate only the frame-dimension subscript to Python __getitem__.
                    first = idx{3};
                    pyIndex = obj.mwToPyIndex(first);

                    % Fetch from Python (NumPy ndarray expected).
                    %pyArr = obj.pyobj.('__getitem__')(pyIndex);
                    %pyGetItem = py.getattr(obj.pyobj, '__getitem__');
                    pyArr = obj.pyGetItem(pyIndex);
                    %pyArr = py.einops.rearrange(pyArr, "N 1 H W -> W H N");
                    %pyArr = datapipes.Ops.py_to_matlab(pyArr);
                    % Convert to MATLAB numeric array when possible.
                    result = MatDatapipes.tensorToMatArray(pyArr);
                    %result = torchTensorToGpuArray(pyArr);

                    result = result(idx{1}, idx{2}, :);

                    % Apply any remaining MATLAB indexing (e.g., dp(5)(:, :, 1))
                    if numel(S) > 1
                        result = builtin('subsref', result, S(2:end));
                    end

                    varargout{1} = result;

                case '.'
                    % Pass-through for method/property access on the wrapper itself
                    % (e.g., dp.shape()) or MATLAB built-ins. We *do not* forward
                    % arbitrary '.' access to the Python object to keep behavior explicit.
                    varargout{1} = builtin('subsref', obj, S);

                case '{}'
                    error('DataPipeWrapper:subsref', 'Brace indexing {} is not supported.');

                otherwise
                    error('DataPipeWrapper:subsref', 'Unsupported indexing type: %s', S(1).type);
            end
        end

        function shp = shape(obj)
            %Converts the underlying datapipes' shape from a torch.Size to a MATLAB shape array
            pyShape = obj.pyobj.shape;
            % torch.Size behaves like a Python tuple
            % Convert to a Python list, then to MATLAB array
            pyList = py.list(pyShape);
        
            % Preallocate MATLAB array
            shp = zeros(1, 3);

            shp(1) = double(pyList(4));
            shp(2) = double(pyList(3));
            shp(3) = double(pyList(1));
            
        end

        function out = or(obj, pipeFunction)
            out = obj.pyobj.pipe_to(pipeFunction);
        end

        function timestampMatArray = getTimestamps(obj) % Blocks until fully cached if RLS
            % block until fully cached
            dataset = py.getattr(obj.getPyHandle(), "_dataset");
            if (py.hasattr(dataset, "block_until_fully_cached"))
                dataset.block_until_fully_cached();
            end
            
            np = py.importlib.import_module("numpy");
            timestampMatArray = uint64(np.asfortranarray(dataset.timestamps));
        end
    end

    methods (Access = private)
        function pyIndex = mwToPyIndex(obj, first)
            % Translate MATLAB first subscript to a Python index acceptable by Datapipe:
            % - scalar -> int (0-based)
            % - ':'    -> slice(0, len)
            % - vector (contiguous, step=1) -> slice(start-1, stop)
            % - otherwise -> error (no step/non-contiguous not supported)
            if ischar(first) && isequal(first, ':')
                % ':' -> full range
                pyIndex = py.slice(py.None);
                return;
            end

            if isnumeric(first)
                first = double(first);
                if isscalar(first)
                    if first < 1 || first > obj.length()
                        error('DataPipeWrapper:Index', 'Index out of bounds (MATLAB is 1-based).');
                    end
                    pyIndex = int64(first - 1);
                    return;
                else
                    % Vectorized indexing: only allow contiguous with unit step.
                    if isempty(first)
                        error('DataPipeWrapper:Index', 'Empty index vector is not allowed.');
                    end
                    % Ensure all are positive integers
                    if any(first < 1) || any(first ~= floor(first))
                        error('DataPipeWrapper:Index', 'Indices must be positive integers.');
                    end
                    % Check contiguity & step=1
                    d = diff(first);
                    if all(d == 1)
                        startM = first(1);
                        stopM  = first(end);      % inclusive in MATLAB
                        if stopM > obj.length()
                            error('DataPipeWrapper:Index', 'Index exceeds number of elements.');
                        end
                        % MATLAB inclusive -> Python exclusive
                        pyIndex = py.slice(int64(startM - 1), int64(stopM));
                    else
                        error(['DataPipeWrapper:Index', ...
                               ' Only contiguous unit-step ranges are supported. ', ...
                               'Use i:j (without a step) or a single index.']);
                    end
                    return;
                end
            end

            % If we reach here, it's not a supported form.
            error('DataPipeWrapper:Index', ...
                  'Unsupported subscript type. Use scalar, ":", or a contiguous i:j range.');
        end

       
    end
end
