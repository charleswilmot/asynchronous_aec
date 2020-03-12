import numpy as np
from copy import deepcopy


class Buffer:
    def __init__(self, size, dtype, batch_size=None):
        self._size = size
        self._dtype = dtype
        self._batch_size = batch_size
        self._buffer = np.zeros(size, dtype=dtype)
        self._write_index = 0
        self._full = False

    def incorporate(self, data):
        remaining = self._size - self._write_index
        data_size = data.shape[0]
        if data_size > remaining:
            self._buffer[self._write_index:] = data[:remaining]
            data_size -= remaining
            self._buffer[:data_size] = data[remaining:]
            self._write_index = data_size
            self._full = True
        else:
            new_write_index = self._write_index + data_size
            self._buffer[self._write_index:new_write_index] = data
            self._write_index = new_write_index

    def get_random_batch(self, batch_size):
        if self._full:
            return self._buffer[np.random.choice(self._size, size=batch_size, replace=False)]
        else:
            if self._write_index > batch_size:
                return self._buffer[np.random.choice(self._write_index, size=batch_size, replace=False)]
            else:
                return self._buffer[:self._write_index]

    def _get_random_batch(self):
        if self._batch_size is not None:
            return self.get_random_batch(self._batch_size)
        else:
            raise ValueError("Buffer must be initialized with a batch size to use this function")

    random_batch = property(_get_random_batch)
