import os
import inspect
from typing import Dict, Tuple, Optional, Callable, List
import importlib.util


def get_breakpoint_hook() -> Callable:
    """
    Get the appropriate debugger hook based on PYTHONBREAKPOINT environment variable.
    Follows the same logic as Python's built-in breakpoint() function.

    Returns:
        Callable: A function that will be called to start the debugger

    Note:
        If PYTHONBREAKPOINT=0, returns a no-op function
        If PYTHONBREAKPOINT is not set, returns pdb.set_trace
        If PYTHONBREAKPOINT specifies a module.function, attempts to load and return it
    """
    breakpoint_env = os.getenv("PYTHONBREAKPOINT", "")

    if breakpoint_env == "0":
        return lambda *args, **kwargs: None

    if "." not in breakpoint_env:
        from pdb import Pdb

        pdb = Pdb()
        return pdb.set_trace

    try:
        module_name, function_name = breakpoint_env.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, function_name)
    except (ImportError, AttributeError) as e:
        import warnings

        warnings.warn(
            f"Couldn't load breakpoint hook '{breakpoint_env}': {e}. "
            "Falling back to pdb.set_trace"
        )
        from pdb import Pdb

        pdb = Pdb()
        return pdb.set_trace
    except Exception as e:
        import warnings

        warnings.warn(
            f"Unexpected error loading breakpoint hook: {e}. "
            "Falling back to pdb.set_trace"
        )
        from pdb import Pdb

        pdb = Pdb()
        return pdb.set_trace


class PdbBreakpoint:
    """
    Represents a breakpoint at a specific location in code.

    Attributes:
        filename (str): Absolute path to the file containing the breakpoint
        lineno (int): Line number of the breakpoint
        is_enabled (bool): Whether the breakpoint is currently active
        skip_count (int): Total number of times to skip when disabled with count
        skip_remaining (int): Remaining number of times to skip
    """

    def __init__(self, filename: str, lineno: int):
        if not os.path.isabs(filename):
            raise ValueError("filename must be an absolute path")
        if not isinstance(lineno, int) or lineno < 1:
            raise ValueError("lineno must be a positive integer")

        self.filename = filename
        self.lineno = lineno
        self.is_enabled = True
        self.skip_count = 0
        self.skip_remaining = 0
        self._frame_cache = None

    def disable(self, count: Optional[int] = None) -> None:
        """
        Disable the breakpoint, optionally for a specified number of hits.

        Args:
            count: If provided, the breakpoint will be re-enabled after being
                  hit this many times while disabled. If None, remains disabled
                  indefinitely.

        Raises:
            ValueError: If count is negative
        """
        if count is not None and count < 0:
            raise ValueError("count must be non-negative")

        self.is_enabled = False
        if count is not None:
            self.skip_count = count
            self.skip_remaining = count
        else:
            self.skip_count = 0
            self.skip_remaining = 0

    def enable(self) -> None:
        """Re-enable the breakpoint and reset skip counters."""
        self.is_enabled = True
        self.skip_remaining = 0

    def check_and_update_status(self) -> bool:
        """
        Check if breakpoint should be active and update its status.

        Returns:
            bool: True if breakpoint should be active, False otherwise
        """
        if self.is_enabled:
            return True

        if self.skip_count == 0:  # Indefinitely disabled
            return False

        self.skip_remaining -= 1
        if self.skip_remaining <= 0:
            self.enable()  # Auto re-enable after count reached
            return True

        return False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PdbBreakpoint):
            return NotImplemented
        return self.filename == other.filename and self.lineno == other.lineno

    def __hash__(self) -> int:
        return hash((self.filename, self.lineno))

    def __str__(self) -> str:
        """String representation in format 'filename:line_number'."""
        return f"{self.filename}:{self.lineno}"


class DawnetPdb:
    """
    Enhanced debugger that supports disabling breakpoints and respects PYTHONBREAKPOINT.

    This class maintains a registry of breakpoints and their states, allowing
    breakpoints to be temporarily disabled or set to auto-enable after a count.
    """

    _breakpoints: Dict[Tuple[str, int], PdbBreakpoint] = {}
    _breakpoint_list: List[PdbBreakpoint] = []
    _debugger_hook: Optional[Callable] = None

    @classmethod
    def _get_debugger_hook(cls) -> Callable:
        """Get (and cache) the debugger hook."""
        if cls._debugger_hook is None:
            cls._debugger_hook = get_breakpoint_hook()
        return cls._debugger_hook

    @classmethod
    def set_trace(cls) -> PdbBreakpoint:
        """
        Set a breakpoint at the current location if enabled.

        Returns:
            PdbBreakpoint: A breakpoint object that can be used to control
                          the breakpoint's behavior

        Note:
            The frame object is properly handled to avoid reference cycles.
        """
        # Get the caller's frame
        frame = inspect.currentframe()
        try:
            if frame is None:
                raise RuntimeError("Could not get current frame")

            caller_frame = frame.f_back
            if caller_frame is None:
                raise RuntimeError("Could not get caller's frame")

            # Get file and line information
            filename = os.path.abspath(caller_frame.f_code.co_filename)
            lineno = caller_frame.f_lineno

            # Create or get existing breakpoint
            location = (filename, lineno)
            if location not in cls._breakpoints:
                breakpoint = PdbBreakpoint(filename, lineno)
                cls._breakpoints[location] = breakpoint
                cls._breakpoint_list.append(breakpoint)
            else:
                breakpoint = cls._breakpoints[location]

            # Check if we should break here
            if breakpoint.check_and_update_status():
                debugger_hook = cls._get_debugger_hook()
                debugger_hook(caller_frame)

            return breakpoint

        finally:
            # Clean up frame references
            del frame
            if "caller_frame" in locals():
                del caller_frame

    @classmethod
    def list_breakpoints(cls) -> None:
        """List all breakpoints with their indices."""
        if not cls._breakpoint_list:
            print("No breakpoints set")
            return

        for idx, bp in enumerate(cls._breakpoint_list):
            status = "enabled" if bp.is_enabled else "disabled"
            print(f"{idx} - {bp} ({status})")

    @classmethod
    def enable_breakpoint(cls, index: Optional[int] = None) -> None:
        """
        Enable breakpoint(s).

        Args:
            index: Index of the breakpoint to enable. If None, enables all breakpoints.

        Raises:
            IndexError: If index is out of range
        """
        if index is None:
            # Enable all breakpoints
            for bp in cls._breakpoint_list:
                bp.enable()
        else:
            try:
                cls._breakpoint_list[index].enable()
            except IndexError:
                raise IndexError(f"No breakpoint at index {index}")

    @classmethod
    def disable_breakpoint(
        cls, index: Optional[int] = None, count: Optional[int] = None
    ) -> None:
        """
        Disable breakpoint(s).

        Args:
            index: Index of the breakpoint to disable. If None, disables all breakpoints.
            count: Optional count of hits before re-enabling (only applies to single breakpoint)

        Raises:
            IndexError: If index is out of range
            ValueError: If applying count to all breakpoints
        """
        if index is None:
            if count is not None:
                raise ValueError("Cannot apply count when disabling all breakpoints")
            # Disable all breakpoints
            for bp in cls._breakpoint_list:
                bp.disable()
        else:
            try:
                cls._breakpoint_list[index].disable(count)
            except IndexError:
                raise IndexError(f"No breakpoint at index {index}")

    @classmethod
    def reset_debugger_hook(cls) -> None:
        """Reset the cached debugger hook to force reloading from environment."""
        cls._debugger_hook = None

    @classmethod
    def clear_all_breakpoints(cls) -> None:
        """Clear all registered breakpoints."""
        cls._breakpoints.clear()
        cls._breakpoint_list.clear()


dpdb = DawnetPdb()
