import numpy as np
import tensorflow as tf
from utils import load_VLCS, eval_accuracy, mini_batch_class_balanced
from models import classificationNN, representationNN
import ot
from scipy.spatial.distance import cdist 
import argparse
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50

parser = argparse.ArgumentParser(description='Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--METHOD', type=str, default="WD", help='WD')
parser.add_argument('--TARGET', type=str, default="3", help='0')
args = parser.parse_args()

METHOD = args.METHOD

SRCS = [0, 2, 3]
TRGS = [int(args.TARGET)]
SRCS.remove(int(args.TARGET))
print(SRCS, TRGS)

CHECKPOINT_PATH = "./checkpoints/vanilla_dg_" + METHOD + "_" + str(TRGS[0])

EPOCHS = 31
BATCH_SIZE = 100
NUM_CLASSES = 5
NUM_DOMAINS = len(SRCS)

HEIGHT = 224
WIDTH = 224
NCH = 3

SAMPLE = 1000
LAMBDA = 1

# Load Dataset
print("Loading data")
src_data, src_labels, _, _, target_data, target_labels = load_VLCS(SRCS, TRGS)

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

X_train = preprocess_input(X_train * 255)
target_images =  preprocess_input(target_images * 255)

Y_train = tf.keras.utils.to_categorical(Y_train, NUM_CLASSES)
target_labels = tf.keras.utils.to_categorical(Y_test, NUM_CLASSES)

D_train = tf.keras.utils.to_categorical(D_train, NUM_DOMAINS)

REP_DIM = 128
base_model = ResNet50(weights='imagenet', include_top=False, pooling="avg")
encoder = representationNN(2048)
classifier = classificationNN(REP_DIM, NUM_CLASSES)

optimizer_base_model = tf.keras.optimizers.Adam(1E-5, beta_1=0.5)
optimizer_encoder = tf.keras.optimizers.Adam(1E-4, beta_1=0.5)
optimizer_logits = tf.keras.optimizers.Adam(1E-4, beta_1=0.5)

ce_loss_none = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

ckpt = tf.train.Checkpoint(base_model = base_model, encoder = encoder, classifier = classifier)
ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=1) 


def L2_dist(x, y):
    dist_1 = tf.reshape(tf.reduce_sum(tf.square(x), 1), [-1, 1])
    dist_2 = tf.reshape(tf.reduce_sum(tf.square(y), 1), [1, -1])
    dist_3 = 2.0 * tf.tensordot(x, tf.transpose(y), axes = 1)  
    return dist_1 + dist_2 - dist_3

@tf.function
def train_step_min_wd(data, class_labels, domainwise_data, domainwise_labels, wasserstein_mappings):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        rep = encoder(base_model(data, training=False), training=True)
        outputs = classifier(rep, training=True)
        loss_cls = tf.reduce_mean(ce_loss_none(class_labels, outputs))
        
        loss_wd_g = 0
        loss_wd_y = 0
        k = 0
        for i in range(NUM_DOMAINS):
            for j in range(i+1, NUM_DOMAINS):
                
                source_rep = encoder(base_model(domainwise_data[i], training=False), training=False) 
                target_rep = encoder(base_model(domainwise_data[j], training=False), training=False)
                
                loss_g_a = L2_dist(source_rep, target_rep)
                
                loss_wd_g += tf.reduce_sum(tf.cast(wasserstein_mappings[k], tf.float32) * loss_g_a)
                
                #loss labels
                loss_y_a = L2_dist(domainwise_labels[i], domainwise_labels[j])
                loss_wd_y += tf.cast(LAMBDA, tf.float32) * tf.reduce_sum(tf.cast(wasserstein_mappings[k], tf.float32) * loss_y_a)
                k += 1
        
        total_loss = loss_cls + 1E-2 * (loss_wd_g + loss_wd_y)

    gradients_basemodel = tape.gradient(total_loss, base_model.trainable_variables)    
    gradients_encoder = tape.gradient(total_loss, encoder.trainable_variables)
    gradients_logits = tape.gradient(total_loss, classifier.trainable_variables)
    
    optimizer_base_model.apply_gradients(zip(gradients_basemodel, base_model.trainable_variables)) 
    optimizer_encoder.apply_gradients(zip(gradients_encoder, encoder.trainable_variables)) 
    optimizer_logits.apply_gradients(zip(gradients_logits, classifier.trainable_variables)) 
    return loss_cls, loss_wd_g, loss_wd_y
    

@tf.function
def train_step_min_erm(data, class_labels):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        rep = encoder(base_model(data, training=False), training=True)
        outputs = classifier(rep, training=True)
        loss = tf.reduce_mean(ce_loss_none(class_labels, outputs))

    gradients_basemodel = tape.gradient(loss, base_model.trainable_variables)    
    gradients_encoder = tape.gradient(loss, encoder.trainable_variables)
    gradients_logits = tape.gradient(loss, classifier.trainable_variables)
    
    optimizer_base_model.apply_gradients(zip(gradients_basemodel, base_model.trainable_variables)) 
    optimizer_encoder.apply_gradients(zip(gradients_encoder, encoder.trainable_variables))
    optimizer_logits.apply_gradients(zip(gradients_logits, classifier.trainable_variables))
    

for epoch in range(EPOCHS):
    
    nb_batches_train = int(len(X_train)/BATCH_SIZE)
    ind_shuf = np.arange(len(X_train))
    np.random.shuffle(ind_shuf)
    
    for batch in range(nb_batches_train):
        ind = mini_batch_class_balanced(D_train, int(BATCH_SIZE/NUM_DOMAINS))
        
        xs = X_train[ind]
        xs = tf.image.pad_to_bounding_box(xs, 4, 4, HEIGHT + 8, WIDTH + 8)
        xs = tf.image.random_crop(xs, [len(xs), HEIGHT, WIDTH, NCH])
        xs = tf.image.random_flip_left_right(xs).numpy()
        
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
                    
                    g_d1 = encoder(base_model(domainwise_data[d1], training=False), training=False).numpy()
                    g_d2 = encoder(base_model(domainwise_data[d2], training=False), training=False).numpy()
                    
                    # distance computation between source and target
                    C0 = cdist(g_d1, g_d2, metric='sqeuclidean')
                    C1 = cdist(domainwise_labels[d1], domainwise_labels[d2], metric='sqeuclidean')
                    
                    C = C0 + LAMBDA * C1
                    
                    wasserstein_mappings.append(ot.emd(ot.unif(g_d1.shape[0]), ot.unif(g_d2.shape[0]), C))
            
            for _ in range(1):
                l1, l2, l3 = train_step_min_wd(xs, ys, domainwise_data, domainwise_labels, wasserstein_mappings)
            
            
    if epoch % 1 == 0:
        print("\nTest Domains:", TRGS, SRCS, METHOD, CHECKPOINT_PATH)
        srcs_train_accuracy, _ = eval_accuracy(X_train, Y_train, base_model, encoder, classifier)
        target_test_accuracy, _ = eval_accuracy(target_images, target_labels, base_model, encoder, classifier)
        print("ERM ec Epoch:", epoch)
        print("Sources:", srcs_train_accuracy)
        print("Target:", target_test_accuracy)
        ckpt_model_save_path = ckpt_manager.save()
        print("\n")
