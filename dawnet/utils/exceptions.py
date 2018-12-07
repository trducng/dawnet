# Exception declaration
# @author: _john
# =============================================================================


class SaveSeparateCheckpointException(Exception):
    """Raise this exception when we want to save model into another checkpoint
    """
    def __init__(self, *args, **kwargs):
        super(SaveSeparateCheckpointException, self).__init__(*args, **kwargs)

class FinishTrainingException(Exception):
    """Raise this exception when finish training"""

    def __init__(self, *args, **kwargs):
        super(FinishTrainingException, self).__init__(*args, **kwargs)
