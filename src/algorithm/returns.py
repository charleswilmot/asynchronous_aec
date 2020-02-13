import numpy as np
import tensorflow as tf


def get_return_func(discount_factor, axis=-1):
    def returns(rewards):
        ret = np.zeros_like(rewards)
        prev = 0
        for i, r in reversed(list(enumerate(rewards))):
            prev = prev * discount_factor + r
            ret[i] = prev
        return ret
    return lambda x: np.apply_along_axis(returns, axis=axis, arr=x)


def tf_returns(rewards, discount_factor, start=None, axis=-1):
    # start must have the same shape than rewards (besides on the 'axis' dim where it should be 1)
    discount_factor = float(discount_factor)
    return_func = get_return_func(discount_factor, axis=axis)
    returns = tf.py_func(return_func, [rewards], rewards.dtype)
    if start is not None and discount_factor != 0:
        rewards_shape = tf.shape(rewards)
        rewards_rank = tf.rank(rewards)
        true_axis = tf.constant(axis)[tf.newaxis] if axis >= 0 else (rewards_rank + axis)[tf.newaxis]
        a = tf.ones(true_axis, dtype=tf.int32)
        b = rewards_shape[true_axis[0], tf.newaxis]
        c = tf.ones(rewards_rank - true_axis - 1, dtype=tf.int32)
        shape = tf.concat([a, b, c], axis=0)
        gammas = tf.fill(shape, discount_factor)
        gammas = tf.cumprod(gammas, axis=axis, reverse=True)
        returns += tf.stop_gradient(start) * gammas
    return returns


if __name__ == "__main__":
    a = tf.constant([[1.0, 2, 3, 4, 5, 6], [1.0, 2, 3, 4, 5, 6]])
    b = tf_returns(a, 0.0, start=[[1000.0], [1001.0]])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(b))
