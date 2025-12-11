from icecream.icecream import IceCreamDebugger, ic, argumentToString, singledispatch, supportTerminalColorsInWindows, highlight, colorize, Source, warnings, callOrValue #_absent, NO_SOURCE_AVAILABLE_WARNING_MESSAGE

def print_shape(arg):
    if hasattr(arg, "shape"):
        # s: torch.Size = arg.shape
        return f"shape={[s for s in arg.shape]}"
    return ""

def print_dtype(arg):
    if hasattr(arg, "dtype"):
        # s: torch.Size = arg.shape
        return f" dtype={arg.dtype}"
    return ""

def add_attr_if_present(arg, attr: str):
    return f", {print_attr(arg, attr)}" if hasattr(arg, attr) else ""
    
def print_attr(arg, attr: str):
    return f"{attr}={getattr(arg, attr)}"
    
def print_type(arg):
    return f"type={type(arg).__name__}"

def print_tensor(arg):
    return f"{print_type(arg)}, {print_shape(arg)}, {print_attr(arg, "dtype")}{add_attr_if_present(arg, "device")}\n\n"

def is_user_defined_instance(arg):
    """
    Check if obj is an instance of a user-defined class (not a built-in type).
    """
    return isinstance(arg, object) and not isinstance(arg, type) and hasattr(arg, "__dict__")


def print_class_instance(arg):
    members_string = "\n\t".join([f"{attr}: {value if value is not None else 'None'}" for attr, value in vars(arg).items()])
    properties_string = "\n\t".join([f"{prop}: {getattr(arg, prop) if getattr(arg, prop) is not None else 'None'}" for prop in dir(arg) if isinstance(getattr(type(arg), prop, None), property)])
    return f"{argumentToString(arg)}\nvariables:\n\t{members_string}\nproperties:\n\t{properties_string}\n\n"

def print_var(arg):
    if hasattr(arg, "shape") and hasattr(arg, "dtype"):
        return print_tensor(arg)
    if is_user_defined_instance(arg):
        return print_class_instance(arg)
    return f"{argumentToString(arg) if argumentToString is not None else ic.argToStringFunction(arg)}, type={type(arg)}"



# sic.__or__ = lambda self, other: self(other)
import inspect

class Sic(IceCreamDebugger):
    """
    Erat Scriptum
    """

    @singledispatch
    def __or__(self, *other):
        return self.sic(*other)
    
    def sic(self, *args, **kwargs):
        if self.enabled:
            argsv = list(args) + list(kwargs.values())
            callFrame = self.get_frame(skip_frames=2)
            self.outputFunction(self._format(callFrame, *argsv))

        if not args:  # E.g. ic().
            passthrough = None
        elif len(args) == 1 and len(kwargs.keys()) == 0:  # E.g. ic(1).
            passthrough = args[0]
        elif len(args) == 0 and len(kwargs.keys()) == 1:  # E.g. ic(1).
            passthrough = kwargs.items()[0]
        else:  # E.g. ic(1, 2, 3).
            passthrough = args, kwargs

        return passthrough

    def __call__(self, *args, **kwargs):
        return self.sic(*args, **kwargs)
    
    # class loop_handler:
    #     def __init__(self, sic, every = 0):
    #         self.sic = sic
    #         self.every = every
    #         self.n = 0

    #     def do_once(self, *args, **kwargs):
    #         """
    #         If running in a loop, call self only in the first iteration
    #         """
    #         if not hasattr(self, '_has_run'):
    #             self._has_run = True
    #             return self.sic(*args, **kwargs)
    #         return args[0] if args else None


    # def dump(self, arg):
    #     # self.sic()
    #     print(print_class_instance(arg))
    # # def once(self):
        
    #     return do_once
    
    def get_frame(self, skip_frames=0):
        """Return the frame after skipping the specified number of frames."""
        callFrame = inspect.currentframe().f_back
        for _ in range(skip_frames):
            if callFrame is not None:
                callFrame = callFrame.f_back
        return callFrame

    def format(self, *args, **kwargs):
        callFrame = inspect.currentframe().f_back
        out = self._format(callFrame, *args, **kwargs)
        return out

    def _format(self, callFrame, *args, **kwargs):
        prefix = callOrValue(self.prefix)

        context = self._formatContext(callFrame)
        if not args:
            time = self._formatTime()
            out = prefix + context + time
        else:
            if not self.includeContext:
                context = ''
            out = self._formatArgs(
                callFrame, prefix, context, args, kwargs=kwargs)

        return out

    def _formatArgs(self, callFrame, prefix, context, args, kwargs=None):
        callNode = Source.executing(callFrame).node
        if callNode is not None:
            source = Source.for_frame(callFrame)
            sanitizedArgStrs = [
                source.get_text_with_indentation(arg)
                for arg in callNode.args]
        else:
            print("No source available for frame:")
            # warnings.warn(
            #     NO_SOURCE_AVAILABLE_WARNING_MESSAGE,
            #     category=RuntimeWarning, stacklevel=4)
            # sanitizedArgStrs = [_absent] * len(args)

        pairs = list(zip(sanitizedArgStrs, args))

        if kwargs is not None:
            pairs += {k: v for k, v in kwargs}

        out = self._constructArgumentOutput(prefix, context, pairs)
        return out

sic = Sic(argToStringFunction=print_var, prefix="sic| ", includeContext=True, outputFunction=lambda s: print(colorize(s)))

