import tensorflow as tf


def calculate_next_power_of_two(number):
    if number < 4:
        return 4
    else:
        pow2 = 4
        while True:
            if number < pow2:
                break
            else:
                pow2 = pow2 ** 2
        return pow2


def build_classification_model(input_shape, n_classes, conv_filters_kernels, dense_units, dropout):
    # Define inputs
    inputs = tf.keras.layers.Input(shape=input_shape)

    """"# Define keras resizing layer to adapt to power of 2 input
    ts_length, n_channels = input_shape
    new_ts_length = calculate_next_power_of_two(ts_length)

    # As resizing only work for images, add Reshape layer to include channel dimension,
    # use resizing and go to original shape
    x = tf.keras.layers.Reshape((ts_length, n_channels, 1))(inputs)
    x = tf.keras.layers.Resizing(new_ts_length, n_channels)(x)
    x = tf.keras.layers.Reshape((new_ts_length, n_channels))(x)"""

    # Define convolutional layers
    for filters, kernel_size in conv_filters_kernels:
        if 'x' not in locals():
            x = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=2, padding="same")(inputs)
        else:
            x = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=2, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

    # Define reshape to dense inputs
    x = tf.keras.layers.Flatten()(x)

    # Define classification head
    for units in dense_units:
        x = tf.keras.layers.Dense(units)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)
        x = tf.keras.layers.Dropout(rate=dropout)(x)

    # Define final classification layer
    outputs = tf.keras.layers.Dense(units=n_classes, activation='softmax')(x)

    # Define model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return model
