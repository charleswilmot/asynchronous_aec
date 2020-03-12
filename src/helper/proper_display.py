class ProperDisplay:
    """Just a helper tool for conveniently showing infos in the terminal during testing phases.
    This class just implements the __repr__ function, which is call when the object is printed.
    """
    def __init__(self, data, i, n):
        self.data = data
        self.i = i
        self.n = n

    def __repr__(self):
        return "chunksize: {} - chunk {}/{}".format(len(self.data), self.i, self.n)