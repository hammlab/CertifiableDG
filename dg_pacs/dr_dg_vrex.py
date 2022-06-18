import numpy as np
import tensorflow as tf
from utils import load_PACS, eval_accuracy,  eval_accuracy_disc, compute_WD, mini_batch_class_balanced
from models import classificationNN, representationNN
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from cleverhans.tf2.attacks.carlini_wagner_l2_small import carlini_wagner_l2 as carlini_wagner_l2_small
import argparse
import time

parser = argparse.ArgumentParser(description='Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--TARGET', type=str, default="0", help='Target domain')
parser.add_argument('--FACTOR', type=str, default="0.5", help='factor of s_adv')
parser.add_argument('--METHOD', type=str, default="VREX", help='VREX')
args = parser.parse_args()

METHOD = args.METHOD
FACTOR = float(args.FACTOR)

SRCS = [0,1,3]
TRGS = [int(args.TARGET)]
SRCS.remove(TRGS[0])

CHECKPOINT_PATH = "./checkpoints/rep_dro_dg_"+METHOD+ "_" + str(TRGS[0]) +"_factor_"+str(FACTOR)

EPOCHS = 11
BATCH_SIZE = 100
NUM_CLASSES = 7
NUM_DOMAINS = len(SRCS)

HEIGHT = 224
WIDTH = 224
NCH = 3

REP_DIM = 128
SAMPLE = 1000
LAMBDA = 1

RHO = 0.1
GAMMA = np.float32(1.)
ADV_STEPS = 50

PRETRAIN_STEPS = 2

# Load Dataset
print("Loading data")
src_data, src_labels, src_test_data, src_test_labels, target_data, target_labels = load_PACS(SRCS, TRGS)

print("Loaded")
D_list = []

for d in range(0, len(src_data)):
    D_list.append(np.ones(len(src_data[d])) * d)

X_train = [item for sublist in src_data for item in sublist]
Y_train = [item for sublist in src_labels for item in sublist]
D_train = [item for sublist in D_list for item in sublist]

X_test_src = [item for sublist in src_test_data for item in sublist]
Y_test_src = [item for sublist in src_test_labels for item in sublist]

X_test_trg = [item for sublist in target_data for item in sublist]
Y_test_trg = [item for sublist in target_labels for item in sublist]


X_train = np.array(X_train, dtype=np.float32).reshape([-1, HEIGHT, WIDTH, NCH])
source_test_images = np.array(X_test_src, dtype=np.float32).reshape([-1, HEIGHT, WIDTH, NCH])
target_images = np.array(X_test_trg, dtype=np.float32).reshape([-1, HEIGHT, WIDTH, NCH])

X_train = preprocess_input(X_train * 255)
source_test_images =  preprocess_input(source_test_images * 255)
target_images =  preprocess_input(target_images * 255)


Y_train = tf.keras.utils.to_categorical(Y_train, NUM_CLASSES)
source_test_labels = tf.keras.utils.to_categorical(Y_test_src, NUM_CLASSES)
target_labels = tf.keras.utils.to_categorical(Y_test_trg, NUM_CLASSES)

D_train = tf.keras.utils.to_categorical(D_train, NUM_DOMAINS)

REP_DIM = 128
base_model = ResNet50(weights='imagenet', include_top=False, pooling="avg")
encoder = representationNN(2048)
classifier = classificationNN(REP_DIM, NUM_CLASSES)

optimizer_base_model = tf.keras.optimizers.Adam(1E-5, beta_1=0.5)
optimizer_encoder = tf.keras.optimizers.Adam(1E-5, beta_1=0.5)
optimizer_logits = tf.keras.optimizers.Adam(1E-3, beta_1=0.5)

optimizer_pretrain_base_model = tf.keras.optimizers.Adam(1E-5, beta_1=0.5)
optimizer_pretrain_encoder = tf.keras.optimizers.Adam(1E-5, beta_1=0.5)
optimizer_pretrain_logits = tf.keras.optimizers.Adam(1E-3, beta_1=0.5)

optimizer_adv = tf.keras.optimizers.Adam(1E-2, beta_1=0.5)

ce_loss_none = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

adv_z = tf.Variable(tf.zeros([BATCH_SIZE, REP_DIM]), trainable=True, name="adv_z")

ckpt = tf.train.Checkpoint(base_model = base_model, encoder = encoder, classifier = classifier)
ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=1) 

@tf.function
def train_step_min_vrex(d1_data, d1_class_labels, d2_data, d2_class_labels, encoded_adv_data, combined_labels):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        d1_rep = encoder(base_model(d1_data, training=False), training=True)
        d1_outputs = classifier(d1_rep, training=True)
        
        d2_rep = encoder(base_model(d2_data, training=False), training=True)
        d2_outputs = classifier(d2_rep, training=True)
        
        d1_loss = tf.reduce_mean(ce_loss_none(d1_class_labels, d1_outputs))
        d2_loss = tf.reduce_mean(ce_loss_none(d2_class_labels, d2_outputs))
        
        adv_outputs = classifier(encoded_adv_data, training=True)
        adv_loss = tf.reduce_mean(ce_loss_none(combined_labels, adv_outputs))
        
        mean_loss = (d1_loss + d2_loss) * 0.5
        penalty = ((d1_loss - mean_loss)**2 + (d2_loss - mean_loss)**2) * 0.5
        
        total_loss = mean_loss + penalty + adv_loss

    gradients_basemodel = tape.gradient(total_loss, base_model.trainable_variables) 
    gradients_encoder = tape.gradient(total_loss, encoder.trainable_variables)
    gradients_logits = tape.gradient(total_loss, classifier.trainable_variables)
    
    optimizer_base_model.apply_gradients(zip(gradients_basemodel, base_model.trainable_variables)) 
    optimizer_encoder.apply_gradients(zip(gradients_encoder, encoder.trainable_variables)) 
    optimizer_logits.apply_gradients(zip(gradients_logits, classifier.trainable_variables))
    
@tf.function
def train_step_min_vrex_pretrain(d1_data, d1_class_labels, d2_data, d2_class_labels):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        d1_rep = encoder(base_model(d1_data, training=False), training=True)
        d1_outputs = classifier(d1_rep, training=True)
        
        d2_rep = encoder(base_model(d2_data, training=False), training=True)
        d2_outputs = classifier(d2_rep, training=True)
        
        d1_loss = tf.reduce_mean(ce_loss_none(d1_class_labels, d1_outputs))
        d2_loss = tf.reduce_mean(ce_loss_none(d2_class_labels, d2_outputs))
        
        mean_loss = (d1_loss + d2_loss) * 0.5
        penalty = ((d1_loss - mean_loss)**2 + (d2_loss - mean_loss)**2) * 0.5
        
        total_loss = mean_loss + penalty

    gradients_basemodel = tape.gradient(total_loss, base_model.trainable_variables)
    gradients_encoder = tape.gradient(total_loss, encoder.trainable_variables)
    gradients_logits = tape.gradient(total_loss, classifier.trainable_variables)
    
    optimizer_pretrain_base_model.apply_gradients(zip(gradients_basemodel, base_model.trainable_variables)) 
    optimizer_pretrain_encoder.apply_gradients(zip(gradients_encoder, encoder.trainable_variables)) 
    optimizer_pretrain_logits.apply_gradients(zip(gradients_logits, classifier.trainable_variables)) 
    
    
      
@tf.function
def train_step_adv_z(encoded_data_adv, encoded_data_orig, labels, GAMMA):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        adv_z.assign(encoded_data_adv)
        
        rep_orig = encoded_data_orig
        distortion = tf.reduce_sum(tf.square(adv_z - rep_orig), axis = 1)
        
        adv_logits = classifier(adv_z, training=False)
        
        adv_loss = ce_loss_none(labels, adv_logits)
        
        loss = -(adv_loss - tf.cast(GAMMA, tf.float32) * distortion)
      
    gradients_adv = tape.gradient(loss, adv_z)
    optimizer_adv.apply_gradients(zip([gradients_adv], [adv_z]))


encoded_X_adv = np.zeros([len(X_train), REP_DIM])
for epoch in range(EPOCHS):
    
    start_time = time.time()
    nb_batches_train = int(len(X_train)/BATCH_SIZE)
    
    for batch in range(nb_batches_train):
        ind = mini_batch_class_balanced(D_train, int(BATCH_SIZE/NUM_DOMAINS))
        
        xs = np.array(X_train[ind])
        ys = np.array(Y_train[ind])
        ds = np.array(D_train[ind])
        
        domain_indices = [np.argwhere(np.argmax(ds, 1) == i).flatten() for i in range(NUM_DOMAINS)]
       
        domainwise_data = []
        domainwise_labels = []
        
        for d1 in range(NUM_DOMAINS):
            ind_d1 = domain_indices[d1]
            domainwise_data.append(xs[ind_d1])
            domainwise_labels.append(ys[ind_d1])
        
        
        if epoch < PRETRAIN_STEPS:
            train_step_min_vrex_pretrain(tf.convert_to_tensor(domainwise_data[0], tf.float32), tf.convert_to_tensor(domainwise_labels[0], tf.float32), 
                                tf.convert_to_tensor(domainwise_data[1], tf.float32), tf.convert_to_tensor(domainwise_labels[1], tf.float32))
                
        else:
            encoded_xs = encoder(base_model(xs, training=False), training=False).numpy()
         
            encoded_xs_adv = np.array(encoded_xs)
            for _ in range(ADV_STEPS):
                
                train_step_adv_z(tf.convert_to_tensor(encoded_xs_adv, tf.float32), tf.convert_to_tensor(encoded_xs, tf.float32), tf.convert_to_tensor(ys, tf.float32), tf.convert_to_tensor(GAMMA, tf.float32))
                
                encoded_xs_adv = adv_z.numpy()
                
                distortion_batch = tf.reduce_sum(tf.square(encoded_xs - encoded_xs_adv), axis = 1)
                
                grad_GAMMA = RHO**2 - np.mean(distortion_batch.numpy())
                GAMMA = tf.clip_by_value(GAMMA - 0.001 * grad_GAMMA, 1E-5, 1E3).numpy()
                
            
            train_step_min_vrex(tf.convert_to_tensor(domainwise_data[0], tf.float32), tf.convert_to_tensor(domainwise_labels[0], tf.float32), 
                            tf.convert_to_tensor(domainwise_data[1], tf.float32), tf.convert_to_tensor(domainwise_labels[1], tf.float32),
                            tf.convert_to_tensor(encoded_xs_adv, tf.float32), ys)
        
        
    if epoch % 1 == 0:
        
        print("\nTest Domains:", TRGS, SRCS, METHOD, CHECKPOINT_PATH, time.time() - start_time)
        srcs_test_accuracy, _ = eval_accuracy(source_test_images, source_test_labels, base_model, encoder, classifier)
        target_test_accuracy, _ = eval_accuracy(target_images, target_labels, base_model, encoder, classifier)
        print("ERM ec Epoch:", epoch)
        print("Sources:", srcs_test_accuracy)
        print("Target:", target_test_accuracy)
        ckpt_model_save_path = ckpt_manager.save()
        
    if epoch >= PRETRAIN_STEPS:    
    
        encoded_source_adv_test_accuracy, encoded_source_adv_test_loss = eval_accuracy_disc(encoded_xs_adv, ys, classifier)
        print("RHO:", RHO, "GAMMA:", GAMMA, "GRAD GAMMA:", grad_GAMMA)
        print("Distortion:", np.mean(distortion_batch.numpy()), np.max(distortion_batch.numpy()), np.min(distortion_batch.numpy()),  len(np.argwhere(distortion_batch.numpy()<2*RHO**2).flatten()))
        print("adv batch acc:", encoded_source_adv_test_accuracy, encoded_source_adv_test_loss)
        
    if epoch % 2 == 0:   
        src_idx = mini_batch_class_balanced(Y_train, sample_size=int(SAMPLE/NUM_CLASSES))
        trg_idx = mini_batch_class_balanced(target_labels, sample_size=int(SAMPLE/NUM_CLASSES))
        src_train_images = X_train[src_idx]
        src_train_labels = Y_train[src_idx]
        trg_test_images = target_images[trg_idx]
        trg_test_labels = target_labels[trg_idx]
        
            
        
        nb_batches_train = int(len(src_train_images)/BATCH_SIZE)
        if len(src_train_images)%BATCH_SIZE!=0:
            nb_batches_train+=1
        for batch in range(nb_batches_train):
            ind_batch = range(BATCH_SIZE * batch, min(BATCH_SIZE * (1+batch), len(src_train_images)))
            if batch == 0:
                encoded_src_train_images = encoder(base_model(src_train_images[ind_batch], training=False), training=False).numpy()
            else:
                encoded_src_train_images = np.concatenate([encoded_src_train_images, encoder(base_model(src_train_images[ind_batch], training=False), training=False).numpy()])

        nb_batches_target = int(len(trg_test_images)/BATCH_SIZE)
        if len(trg_test_images)%BATCH_SIZE!=0:
            nb_batches_target+=1
        for batch in range(nb_batches_target):
            ind_batch = range(BATCH_SIZE * batch, min(BATCH_SIZE * (1+batch), len(trg_test_images)))
            if batch == 0:
                encoded_trg_test_images = encoder(base_model(trg_test_images[ind_batch], training=False), training=False).numpy()
            else:
                encoded_trg_test_images = np.concatenate([encoded_trg_test_images, encoder(base_model(trg_test_images[ind_batch], training=False), training=False).numpy()])
                
        encoded_src_train_images_adv = carlini_wagner_l2_small(classifier, encoded_src_train_images, clip_min=np.min(encoded_src_train_images), clip_max=np.max(encoded_src_train_images), 
        binary_search_steps=5,  max_iterations=50, batch_size=200, learning_rate=1E-1)
        cw_acc, cw_loss = eval_accuracy_disc(encoded_src_train_images_adv, src_train_labels, classifier)
        print("Attack success:", 100 - cw_acc, cw_loss)
        print("Attack distortion:", tf.reduce_mean(tf.reduce_sum(tf.square(encoded_src_train_images_adv - encoded_src_train_images), axis = 1)).numpy())
        WDs_sadv = compute_WD(encoded_src_train_images, src_train_labels, encoded_src_train_images_adv, src_train_labels)
        WDs_target = compute_WD(encoded_src_train_images, src_train_labels, encoded_trg_test_images, trg_test_labels)
        RHO = FACTOR * WDs_sadv
        print("updating RHO:", RHO)
        print(WDs_sadv, "RHO_target:", WDs_target/WDs_sadv)
        
        
        
        