# The review functions
# @author: _john
# ==============================================================================

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name):
        """Initialize the helper"""
        self.reset()
        self.name = name

    def reset(self):
        """Reset the helper"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update the helper"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
