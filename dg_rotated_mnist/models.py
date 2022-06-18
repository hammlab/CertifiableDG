import tensorflow as tf
import tensorflow_addons as tfa 

def domain_predictor(inputs, num_classes, input_size):
    inputs = tf.keras.layers.Input(shape=(input_size), name='inputs')
    
    x = inputs
    
    x = tf.keras.layers.Dense(100)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(100)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(num_classes)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

def domain_predictor_cdan(inputs, num_classes, input_size):
    inputs = tf.keras.layers.Input(shape=(input_size), name='inputs')
    
    x = inputs
    
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(num_classes)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

def representationNN(input_shape):
    
    inputs = tf.keras.layers.Input(shape=input_shape[1:], name='inputs')
    
    x = inputs
    x = tf.keras.layers.Conv2D(32, (3, 3))(x)
    x = tfa.layers.GroupNormalization(groups=8, axis=3)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(2,2))(x)
    x = tfa.layers.GroupNormalization(groups=8, axis=3)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Conv2D(128, (3, 3))(x)
    x = tfa.layers.GroupNormalization(groups=8, axis=3)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Conv2D(128, (3, 3))(x)
    x = tfa.layers.GroupNormalization(groups=8, axis=3)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x)


def classificationNN(rep_dim, num_classes):
    inputs = tf.keras.layers.Input(shape=(rep_dim), name='inputs')
    
    x = inputs
    x = tf.keras.layers.Dense(num_classes)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)