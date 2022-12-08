"""
father of all loss method.
"""

class BaseLoss(object):
    """
    builder losss.
    """
    def __init__(self, cfg):
        """
        init

        """
        self.cfg = cfg

    def __call__(self, **kwargs):
        """
        call method
        """
        pass

