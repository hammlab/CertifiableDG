import numpy as np
import tensorflow as tf
from utils import load_rotated_mnist, eval_accuracy,  eval_accuracy_disc, compute_WD, mini_batch_class_balanced
from models import representationNN, classificationNN, representationNN_small
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2 as carlini_wagner_l2
from cleverhans.tf2.attacks.carlini_wagner_l2_small import carlini_wagner_l2 as carlini_wagner_l2_small
from cleverhans.tf2.attacks.carlini_wagner_rep_l2 import carlini_wagner_rep_l2
import ot
from scipy.spatial.distance import cdist 
import argparse

parser = argparse.ArgumentParser(description='Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--TARGET', type=str, default="2", help='Target domain')
parser.add_argument('--METHOD', type=str, default="VREX", help='VREX')
args = parser.parse_args()

METHOD = args.METHOD

CHECKPOINT_PATH = "./checkpoints/vanilla_dg_"+METHOD

SRCS = [0,1,2]
TRGS = [int(args.TARGET)]
SRCS.remove(TRGS[0])

EPOCHS = 501
BATCH_SIZE = 200
NUM_CLASSES = 10
NUM_DOMAINS = len(SRCS)

HEIGHT = 28
WIDTH = 28
NCH = 1

REP_DIM = 128
SAMPLE = 1000
LAMBDA = 1

# Load Dataset
print("Loading data")
src_data, src_labels, _, _, target_data, target_labels, _, _ = load_rotated_mnist(SRCS, TRGS)

print("Loaded")
D_list = []

for d in range(0, len(src_data)):
    D_list.append(np.ones(len(src_data[d])) * d)

X_train = [item for sublist in src_data for item in sublist]
Y_train = [item for sublist in src_labels for item in sublist]
D_train = [item for sublist in D_list for item in sublist]

X_test = [item for sublist in target_data for item in sublist]
Y_test = [item for sublist in target_labels for item in sublist]

X_train = np.array(X_train, dtype=np.float32).reshape([-1, HEIGHT, WIDTH, NCH])
target_images = np.array(X_test, dtype=np.float32).reshape([-1, HEIGHT, WIDTH, NCH])

Y_train = tf.keras.utils.to_categorical(Y_train, NUM_CLASSES)
target_labels = tf.keras.utils.to_categorical(Y_test, NUM_CLASSES)

D_train = tf.keras.utils.to_categorical(D_train, NUM_DOMAINS)

print(D_train.shape, X_train.shape, Y_train.shape)
print(target_images.shape, target_labels.shape)
print(np.min(X_train), np.max(X_train))

encoder = representationNN(X_train.shape)

classifier = classificationNN(REP_DIM, NUM_CLASSES)

optimizer_encoder = tf.keras.optimizers.Adam(1E-3, beta_1=0.5)
optimizer_logits = tf.keras.optimizers.Adam(1E-3, beta_1=0.5)

ce_loss_none = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

ckpt = tf.train.Checkpoint(encoder = encoder, classifier = classifier)
ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=1) 

@tf.function
def train_step_min_vrex(d1_data, d1_class_labels, d2_data, d2_class_labels):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        d1_rep = encoder(d1_data, training=True)
        d1_outputs = classifier(d1_rep, training=True)
        
        d2_rep = encoder(d2_data, training=True)
        d2_outputs = classifier(d2_rep, training=True)
        
        d1_loss = tf.reduce_mean(ce_loss_none(d1_class_labels, d1_outputs))
        d2_loss = tf.reduce_mean(ce_loss_none(d2_class_labels, d2_outputs))
        
        mean_loss = (d1_loss + d2_loss) * 0.5
        penalty = ((d1_loss - mean_loss)**2 + (d2_loss - mean_loss)**2) * 0.5
        
        loss = mean_loss + penalty

    gradients_encoder = tape.gradient(loss, encoder.trainable_variables)
    gradients_logits = tape.gradient(loss, classifier.trainable_variables)
    
    optimizer_encoder.apply_gradients(zip(gradients_encoder, encoder.trainable_variables))
    optimizer_logits.apply_gradients(zip(gradients_logits, classifier.trainable_variables))
    

for epoch in range(EPOCHS):
    
    nb_batches_train = int(len(X_train)/BATCH_SIZE)
    ind_shuf = np.arange(len(X_train))
    np.random.shuffle(ind_shuf)
    
    for batch in range(nb_batches_train):
       
        ind = mini_batch_class_balanced(D_train, int(BATCH_SIZE/NUM_DOMAINS))
        
        xs = X_train[ind]
        ys = Y_train[ind]
        ds = D_train[ind]
    
        domain_indices = [np.argwhere(np.argmax(ds, 1) == i).flatten() for i in range(NUM_DOMAINS)]
       
        domainwise_data = []
        domainwise_labels = []
        
        for d1 in range(NUM_DOMAINS):
            ind_d1 = domain_indices[d1]
            domainwise_data.append(xs[ind_d1])
            domainwise_labels.append(ys[ind_d1])
        
        for _ in range(1):
            train_step_min_vrex(tf.convert_to_tensor(domainwise_data[0], tf.float32), tf.convert_to_tensor(domainwise_labels[0], tf.float32), 
                                tf.convert_to_tensor(domainwise_data[1], tf.float32), tf.convert_to_tensor(domainwise_labels[1], tf.float32))
                
            
    if epoch % 100 == 0:
        print("\nTest Domains:", TRGS, SRCS, METHOD, CHECKPOINT_PATH)
        srcs_train_accuracy, _ = eval_accuracy(X_train, Y_train, encoder, classifier)
        target_test_accuracy, _ = eval_accuracy(target_images, target_labels, encoder, classifier)
        print("Vrex ec Epoch:", epoch)
        print("Sources:", srcs_train_accuracy)
        print("Target:", target_test_accuracy)
        ckpt_model_save_path = ckpt_manager.save()
        print("\n")