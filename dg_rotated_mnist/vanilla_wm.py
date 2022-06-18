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
parser.add_argument('--METHOD', type=str, default="ERM", help='ERM or WD')
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

encoder = representationNN(X_train.shape)
classifier = classificationNN(REP_DIM, NUM_CLASSES)

optimizer_encoder = tf.keras.optimizers.Adam(1E-3, beta_1=0.5)
optimizer_logits = tf.keras.optimizers.Adam(1E-3, beta_1=0.5)

ce_loss_none = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

ckpt = tf.train.Checkpoint(encoder = encoder, classifier = classifier)
ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=1) 

def L2_dist(x, y):
    '''
    compute the squared L2 distance between two matrics
    '''
    dist_1 = tf.reshape(tf.reduce_sum(tf.square(x), 1), [-1, 1])
    dist_2 = tf.reshape(tf.reduce_sum(tf.square(y), 1), [1, -1])
    dist_3 = 2.0 * tf.tensordot(x, tf.transpose(y), axes = 1) 
    return dist_1 + dist_2 - dist_3

@tf.function
def train_step_min_wd(data, class_labels, domainwise_data, domainwise_labels, wasserstein_mappings):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        rep = encoder(data, training=True)
        outputs = classifier(rep, training=True)
        loss_cls = tf.reduce_mean(ce_loss_none(class_labels, outputs))
        
        loss_wd_g = 0
        loss_wd_y = 0
        k = 0
        for i in range(NUM_DOMAINS):
            for j in range(i+1, NUM_DOMAINS):
                
                source_rep = encoder(domainwise_data[i], training=True)
                target_rep = encoder(domainwise_data[j], training=True)
                
                loss_g_a = L2_dist(source_rep, target_rep)
                loss_wd_g += tf.reduce_sum(tf.cast(wasserstein_mappings[k], tf.float32) * loss_g_a)
                
                #loss labels
                loss_y_a = L2_dist(domainwise_labels[i], domainwise_labels[j])
                loss_wd_y += tf.cast(LAMBDA, tf.float32) * tf.reduce_sum(tf.cast(wasserstein_mappings[k], tf.float32) * loss_y_a)
                k += 1
        
        total_loss = loss_cls + 0.1 * ( loss_wd_g + loss_wd_y)
        
    gradients_encoder = tape.gradient(total_loss, encoder.trainable_variables)
    gradients_logits = tape.gradient(total_loss, classifier.trainable_variables)
    
    optimizer_encoder.apply_gradients(zip(gradients_encoder, encoder.trainable_variables)) 
    optimizer_logits.apply_gradients(zip(gradients_logits, classifier.trainable_variables)) 
    return loss_cls, loss_wd_g, loss_wd_y
    

@tf.function
def train_step_min_erm(data, class_labels):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        rep = encoder(data, training=True)
        outputs = classifier(rep, training=True)
        loss = tf.reduce_mean(ce_loss_none(class_labels, outputs))

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
        
        if METHOD == 'ERM':
            for _ in range(1):
                train_step_min_erm(xs, ys)
                
        else:
            ds = D_train[ind]
        
            domain_indices = [np.argwhere(np.argmax(ds, 1) == i).flatten() for i in range(NUM_DOMAINS)]
           
            domainwise_data = []
            domainwise_labels = []
            
            for d1 in range(NUM_DOMAINS):
                ind_d1 = domain_indices[d1]
                domainwise_data.append(xs[ind_d1])
                domainwise_labels.append(ys[ind_d1])
            
            wasserstein_mappings = []
            for d1 in range(NUM_DOMAINS):
                for d2 in range(d1+1, NUM_DOMAINS):
                    
                    g_d1 = encoder(domainwise_data[d1], training=False).numpy()
                    g_d2 = encoder(domainwise_data[d2], training=False).numpy()
                    
                    # distance computation between source and target
                    C0 = cdist(g_d1, g_d2, metric='sqeuclidean')
                    C1 = cdist(domainwise_labels[d1], domainwise_labels[d2], metric='sqeuclidean')
                    
                    C = C0 + LAMBDA * C1
                    
                    wasserstein_mappings.append(ot.emd(ot.unif(g_d1.shape[0]), ot.unif(g_d2.shape[0]), C))
            
            
            for _ in range(1):
                l1, l2, l3 = train_step_min_wd(xs, ys, domainwise_data, domainwise_labels, wasserstein_mappings)
            
            
    if epoch % 100 == 0:
        print("\nTest Domains:", TRGS, SRCS, METHOD, CHECKPOINT_PATH)
        srcs_train_accuracy, _ = eval_accuracy(X_train, Y_train, encoder, classifier)
        target_test_accuracy, _ = eval_accuracy(target_images, target_labels, encoder, classifier)
        print("WM ec Epoch:", epoch)
        print("Sources:", srcs_train_accuracy)
        print("Target:", target_test_accuracy)
        ckpt_model_save_path = ckpt_manager.save()
        print("\n")