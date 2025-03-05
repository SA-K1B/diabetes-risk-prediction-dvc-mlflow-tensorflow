"""Module for building a diabetes risk prediction model using TensorFlow."""
import tensorflow as tf


def build_model():
    """Builds and compiles a TensorFlow Keras model for diabetes risk prediction."""
    model = tf.keras.Sequential([  # Use tf.keras instead of keras directly
        tf.keras.layers.Dense(4, activation='sigmoid', input_shape=(8,)),
        tf.keras.layers.Dense(2, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


# Call the build_model function
# model = build_model()
# print(model.summary())
