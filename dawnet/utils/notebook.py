# Utility functions for Google colab support
# @author: _john
# =============================================================================
import os
import multiprocessing


def run_in_process(f, *args, **kwargs):
    """Run the function inside other process.

    Useful not to block cell inside notebook.

    # Args
        f <function>: the function to run inside worker process
        args <()>: tuple of parameters to pass into that function

    # Returns
        <{}>: the output result, can be accessed with 'result'
        <Process>: the process, in case you want to set `.join()`
    """
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    def wrapper():
        pid = os.getpid()
        print(f'Running process {pid}')
        value = f(*args, **kwargs)
        return_dict['result'] = value
        print(f'Finished process {pid}')

    p = multiprocessing.Process(target=wrapper, args=(), daemon=True)
    p.start()

    return return_dict, p


def is_ready(return_dict):
    """Check if the process result is ready"""
    if isinstance(return_dict, multiprocessing.managers.DictProxy):
        if return_dict.get('result') is not None:
            return True

    return False


def gpu_info():
    """Get GPU information"""
    # subprocess.run(['ln', '-sf', '/opt/bin/nvidia-smi', '/usr/bin/nvidia-smi'])
    # subprocess.run(['pip', 'install', 'gputil'])
    # subprocess.run(['pip', 'install', 'psutil'])
    # subprocess.run(['pip', 'install', 'humanize'])

    import psutil, humanize, os
    import GPUtil as GPU

    gpu = GPU.getGPUs()[0]
    process = psutil.Process(os.getpid())
    print("Gen RAM Free: {} | Proc size: {}".format(
        humanize.naturalsize(psutil.virtual_memory().available),
        humanize.naturalsize(process.memory_info().rss)))
    print("GPU RAM Free: {0:.0f}MB | "
          "Used: {1:.0f}MB | "
          "Util {2:3.0f}% | "
          "Total {3:.0f}MB".format(
        gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))


def is_notebook() -> bool:
    """Check if the program is running inside a notebook

    Credits: https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    """  # noqa: E501
    try:
        shell = get_ipython().__class__.__name__    # type: ignore
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


is_in_notebook: bool = is_notebook()
