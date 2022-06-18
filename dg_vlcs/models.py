import tensorflow as tf


def domain_predictor(num_classes, input_size):
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

def domain_predictor_cdan(num_classes, input_size):
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

def representationNN(rep_dim):
    
    inputs = tf.keras.layers.Input(shape=(rep_dim), name='inputs')
    
    x = inputs
    
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(128)(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def classificationNN(rep_dim, num_classes):
    inputs = tf.keras.layers.Input(shape=(rep_dim), name='inputs')
    
    x = inputs
    x = tf.keras.layers.Dense(num_classes)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


    