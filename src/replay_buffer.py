import numpy as np


class Buffer:
    def __init__(self, size):
        self._size = size
        self._current_size = 0
        self._data = []

    def incorporate_multiple(self, iterator):
        for element in iterator:
            self.incorporate(element)

    def incorporate(self, element):
        if self._current_size < self._size:
            self._data.append(element)
            self._current_size += 1
        else:
            self._data[np.random.randint(0, self._current_size)] = element

    def batch(self, size):
        if size >= self._current_size:
            return self._data
        else:
            indices = np.random.choice(self._current_size, size, replace=False)
            return [self._data[i] for i in indices]
