import matplotlib.pyplot as plt


class FigureManager:
    _path = "./"
    _save = True

    def __init__(self, filename):
        self._fig = plt.figure(dpi=200)
        self._filename = filename

    def __enter__(self):
        return self._fig

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if FigureManager._save:
            print("saving plot {}  ...  ".format(self._filename), end="")
            self._fig.savefig(FigureManager._path + self._filename)
            print("done")
            plt.close(self._fig)
        else:
            plt.show()
