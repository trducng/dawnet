# Utility functions for Google colab support
# @author: _john
# =============================================================================
import subprocess


def gpu_info():
    """Get GPU information"""
    subprocess.run(['ln', '-sf', '/opt/bin/nvidia-smi', '/usr/bin/nvidia-smi'])
    subprocess.run(['pip', 'install', 'gputil'])
    subprocess.run(['pip', 'install', 'psutil'])
    subprocess.run(['pip', 'install', 'humanize'])

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
