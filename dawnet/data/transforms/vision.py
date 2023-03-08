class GetBands(object):
    """
    Gets the first X bands of the tile.
    """

    def __init__(self, bands):
        assert bands >= 0, "Must get at least 1 band"
        self.bands = bands

    def __call__(self, x):
        # Tiles are already in [c, w, h] order
        return x[: self.bands, :, :]
