import numpy as np
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
                pow2 = pow2 * 2
        return pow2


class ClassificationModelConstructorV1:
    def __init__(self, input_shape, n_classes, dropout):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.dropout = dropout
        self.cnn_simple_arch = {"conv_filters_kernels":  [(32, 5), (64, 5)], 'dense_units': [64]}
        self.cnn_intermediate_arch = {"conv_filters_kernels":  [(16, 5), (32, 5), (64, 3)], 'dense_units': [64]}
        self.cnn_complex_arch = {"conv_filters_kernels":  [(16, 5), (32, 5), (64, 5), (128, 3)], 'dense_units': [128]}

    @staticmethod
    def _build_classification_model(input_shape, n_classes, conv_filters_kernels, dense_units, dropout):
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

    def get_model(self, model_name):
        if model_name == "cnn-simple":
            conv_filters_kernels = self.cnn_simple_arch["conv_filters_kernels"]
            dense_units = self.cnn_simple_arch["dense_units"]
        elif model_name == "cnn-intermediate":
            conv_filters_kernels = self.cnn_intermediate_arch["conv_filters_kernels"]
            dense_units = self.cnn_intermediate_arch["dense_units"]
        elif model_name == "cnn-complex":
            conv_filters_kernels = self.cnn_complex_arch["conv_filters_kernels"]
            dense_units = self.cnn_complex_arch["dense_units"]
        else:
            raise NameError(f"Model name {model_name} is not valid.")

        model = self._build_classification_model(input_shape=self.input_shape, n_classes=self.n_classes,
                                                 conv_filters_kernels=conv_filters_kernels, dense_units=dense_units,
                                                 dropout=self.dropout)

        return model


class AEModelConstructorV1:
    def __init__(self, input_shape, stride, compression_rate):
        self.input_shape = input_shape
        self.stride = stride
        self.compression_rate = compression_rate
        self.ae_cnn_shallow_arch = {"encoder_filters_kernels": [(16, 7)]}
        self.ae_cnn_simple_arch = {"encoder_filters_kernels":  [(16, 7), (32, 5)]}
        self.ae_cnn_intermediate_arch = {"encoder_filters_kernels":  [(16, 7), (32, 5), (64, 3)]}
        self.ae_cnn_complex_arch = {"encoder_filters_kernels":  [(16, 7), (32, 5), (64, 3), (128, 3)]}

    @staticmethod
    def _build_ae_model(input_shape, compression_rate, encoder_filters_kernels, stride):
        # Define inputs
        inputs = tf.keras.layers.Input(shape=input_shape)

        # Define keras resizing layer to adapt to power of 2 input
        ts_length, n_channels = input_shape
        new_ts_length = calculate_next_power_of_two(ts_length)
        total_input_size = new_ts_length * n_channels

        """# As resizing only work for images, add Reshape layer to include channel dimension,
        # use resizing and go to original shape
        x = tf.keras.layers.Reshape((ts_length, n_channels, 1))(inputs)
        x = tf.keras.layers.Resizing(new_ts_length, n_channels)(x)
        x = tf.keras.layers.Reshape((new_ts_length, n_channels))(x)"""

        # Add zero padding to input to get a power of 2 in the temporal dimension
        padding_count = new_ts_length - ts_length
        if padding_count % 2 == 0:
            x = tf.keras.layers.ZeroPadding1D(padding=padding_count // 2)(inputs)
        else:
            x = tf.keras.layers.ZeroPadding1D(padding=(padding_count // 2 + 1, padding_count // 2))(inputs)

        # Define encoder convolutional layers
        for filters, kernel_size in encoder_filters_kernels:
            x = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=stride, padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)

        """# Check that there is compression in the latent space
        reduction_rate = (x.shape[1] * x.shape[2]) / (ts_length * n_channels)
        if reduction_rate > 0.6:
            raise ValueError("There is no enough compression in the AE")
        else:
            print(f"Reduction rate: {reduction_rate:.2f}; ({ts_length}, {n_channels}) -> ({x.shape[1]}, {x.shape[2]})")"""

        # Add bottleneck layer
        latent_length, latent_channels = x.shape[1], x.shape[2]
        new_latent_channels = np.floor(compression_rate * total_input_size / latent_length)
        x = tf.keras.layers.Conv1D(filters=new_latent_channels, kernel_size=3, strides=1, padding="same")(x)

        """bottleneck_size = np.floor(total_input_size * compression_rate)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(bottleneck_size, activation='relu')(x)
        
        # Define decoder dense layer
        x = tf.keras.layers.Dense(latent_length * latent_channels, activation='relu')(x)
        x = tf.keras.layers.Reshape((latent_length, latent_channels))(x)"""

        x = tf.keras.layers.Conv1D(filters=latent_channels, kernel_size=3, strides=1, padding="same")(x)

        # Define decoder convolutional layers
        for filters, kernel_size in encoder_filters_kernels[::-1]:
            x = tf.keras.layers.Conv1DTranspose(filters=filters, kernel_size=kernel_size, strides=stride,
                                                padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)

        # Output layer: Conv layer + cropping
        decoder_outputs = tf.keras.layers.Conv1D(filters=n_channels, kernel_size=3, padding="same")(x)
        if padding_count % 2 == 0:
            decoder_outputs = tf.keras.layers.Cropping1D(cropping=padding_count // 2)(decoder_outputs)
        else:
            decoder_outputs = tf.keras.layers.Cropping1D(cropping=(padding_count // 2 + 1, padding_count // 2))(
                decoder_outputs)

        # Define model
        model = tf.keras.models.Model(inputs=inputs, outputs=decoder_outputs)

        return model

    def get_model(self, model_name):
        if model_name == "ae-cnn-shallow":
            encoder_filters_kernels = self.ae_cnn_shallow_arch["encoder_filters_kernels"]
        elif model_name == "ae-cnn-simple":
            encoder_filters_kernels = self.ae_cnn_simple_arch["encoder_filters_kernels"]
        elif model_name == "ae-cnn-intermediate":
            encoder_filters_kernels = self.ae_cnn_intermediate_arch["encoder_filters_kernels"]
        elif model_name == "ae-cnn-complex":
            encoder_filters_kernels = self.ae_cnn_complex_arch["encoder_filters_kernels"]
        else:
            raise NameError(f"Model name {model_name} is not valid.")

        model = self._build_ae_model(self.input_shape, self.compression_rate, encoder_filters_kernels, self.stride)

        return model
