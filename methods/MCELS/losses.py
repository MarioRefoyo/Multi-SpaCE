import tensorflow as tf


def tv_norm(signal, tv_beta):
    flat_signal = signal.flatten()
    signal_grad = flat_signal[:-1] - flat_signal[1:]
    return signal_grad.abs().pow(tv_beta).mean()


def tv_norm_tf(signal, tv_beta):
    flat_signal = tf.reshape(signal, [-1])
    signal_grad = flat_signal[:-1] - flat_signal[1:]
    return tf.reduce_mean(tf.abs(signal_grad) ** tv_beta)
