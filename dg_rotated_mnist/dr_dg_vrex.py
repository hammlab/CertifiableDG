import numpy as np
import tensorflow as tf
from utils import load_rotated_mnist, eval_accuracy,  eval_accuracy_disc, compute_WD, mini_batch_class_balanced
from models import representationNN, classificationNN
from cleverhans.tf2.attacks.carlini_wagner_l2_small import carlini_wagner_l2 as carlini_wagner_l2_small
import argparse

parser = argparse.ArgumentParser(description='Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--TARGET', type=str, default="2", help='Target domain')
parser.add_argument('--METHOD', type=str, default="VREX", help='VREX')
parser.add_argument('--FACTOR', type=str, default="0.5", help='factor of s_adv')
args = parser.parse_args()

METHOD = args.METHOD
FACTOR = float(args.FACTOR)

CHECKPOINT_PATH = "./checkpoints/rep_dro_dg_"+METHOD+"_factor_"+str(FACTOR)

SRCS = [0,1,2]
TRGS = [int(args.TARGET)]
SRCS.remove(TRGS[0])

EPOCHS = 201
BATCH_SIZE = 200
NUM_CLASSES = 10
NUM_DOMAINS = len(SRCS)

HEIGHT = 28
WIDTH = 28
NCH = 1

REP_DIM = 128
SAMPLE = 1000
LAMBDA = 1

ADV_STEPS = 20
PRETRAIN_STEPS = 20
GAMMA = np.float32(1.)
RHO = 0.1

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

optimizer_pretrain_encoder = tf.keras.optimizers.Adam(1E-3, beta_1=0.5)
optimizer_pretrain_logits = tf.keras.optimizers.Adam(1E-3, beta_1=0.5)

optimizer_encoder = tf.keras.optimizers.Adam(1E-3, beta_1=0.5)
optimizer_logits = tf.keras.optimizers.Adam(1E-3, beta_1=0.5)

ce_loss_none = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

adv_z = tf.Variable(tf.zeros([BATCH_SIZE, REP_DIM]), trainable=True, name="adv_z")
optimizer_adv = tf.keras.optimizers.Adam(1E-2, beta_1=0.5)

full_outputs = classifier(encoder.output)
full_models = tf.keras.Model(inputs=encoder.inputs, outputs=full_outputs)

ckpt = tf.train.Checkpoint(encoder = encoder, classifier = classifier)
ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=1) 


@tf.function
def train_step_min_vrex_pretrain(d1_data, d1_class_labels, d2_data, d2_class_labels):
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
    
    optimizer_pretrain_encoder.apply_gradients(zip(gradients_encoder, encoder.trainable_variables))
    optimizer_pretrain_logits.apply_gradients(zip(gradients_logits, classifier.trainable_variables))
    
@tf.function
def train_step_min_vrex(d1_data, d1_class_labels, d2_data, d2_class_labels, encoded_adv_data, combined_labels):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        d1_rep = encoder(d1_data, training=True)
        d1_outputs = classifier(d1_rep, training=True)
        
        d2_rep = encoder(d2_data, training=True)
        d2_outputs = classifier(d2_rep, training=True)
        
        d1_loss = tf.reduce_mean(ce_loss_none(d1_class_labels, d1_outputs))
        d2_loss = tf.reduce_mean(ce_loss_none(d2_class_labels, d2_outputs))
        
        adv_outputs = classifier(encoded_adv_data, training=True)
        adv_loss = tf.reduce_mean(ce_loss_none(combined_labels, adv_outputs))
        
        mean_loss = (d1_loss + d2_loss) * 0.5
        penalty = ((d1_loss - mean_loss)**2 + (d2_loss - mean_loss)**2) * 0.5
        
        loss = mean_loss + penalty + adv_loss

    gradients_encoder = tape.gradient(loss, encoder.trainable_variables)
    gradients_logits = tape.gradient(loss, classifier.trainable_variables)
    
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
        
        
        if epoch < PRETRAIN_STEPS:
            for _ in range(1):
                train_step_min_vrex_pretrain(tf.convert_to_tensor(domainwise_data[0], tf.float32), tf.convert_to_tensor(domainwise_labels[0], tf.float32), 
                                tf.convert_to_tensor(domainwise_data[1], tf.float32), tf.convert_to_tensor(domainwise_labels[1], tf.float32))
                
        else:
            encoded_xs = encoder(xs, training=False)
            encoded_xs_adv = np.array(encoded_X_adv[ind])
        
            for _ in range(ADV_STEPS):
                
                train_step_adv_z(tf.convert_to_tensor(encoded_xs_adv, tf.float32), tf.convert_to_tensor(encoded_xs, tf.float32), tf.convert_to_tensor(ys, tf.float32), tf.convert_to_tensor(GAMMA, tf.float32))
                
                encoded_xs_adv = adv_z.numpy()
                encoded_X_adv[ind] = encoded_xs_adv
                
                distortion_batch = tf.reduce_sum(tf.square(encoded_xs - encoded_xs_adv), axis = 1)
                
                grad_GAMMA = RHO**2 - np.mean(distortion_batch.numpy())
                GAMMA = tf.clip_by_value(GAMMA - 0.1 * grad_GAMMA, 1E-3, 1E3).numpy()
                
            
            train_step_min_vrex(tf.convert_to_tensor(domainwise_data[0], tf.float32), tf.convert_to_tensor(domainwise_labels[0], tf.float32), 
                            tf.convert_to_tensor(domainwise_data[1], tf.float32), tf.convert_to_tensor(domainwise_labels[1], tf.float32),
                            tf.convert_to_tensor(encoded_xs_adv, tf.float32), ys)
                
            
    if epoch % 10 == 0:
        print("\nTest Domains:", TRGS, SRCS, METHOD, CHECKPOINT_PATH)
        srcs_train_accuracy, _ = eval_accuracy(X_train, Y_train, encoder, classifier)
        target_test_accuracy, _ = eval_accuracy(target_images, target_labels, encoder, classifier)
        print("VRex Epoch:", epoch)
        print("Sources:", srcs_train_accuracy)
        print("Target:", target_test_accuracy)
        ckpt_model_save_path = ckpt_manager.save()
        
        
    if epoch % 10 == 0 and epoch > PRETRAIN_STEPS:    
        encoded_source_adv_test_accuracy, encoded_source_adv_test_loss = eval_accuracy_disc(encoded_xs_adv, ys, classifier)
        print("RHO:", RHO, "GAMMA:", GAMMA, "GRAD GAMMA:", grad_GAMMA)
        print("Distortion:", np.mean(distortion_batch.numpy()), np.max(distortion_batch.numpy()), np.min(distortion_batch.numpy()),  len(np.argwhere(distortion_batch.numpy()<2*RHO**2).flatten()))
        print("adv batch acc:", encoded_source_adv_test_accuracy, encoded_source_adv_test_loss)
        
        
        src_idx = mini_batch_class_balanced(X_train, sample_size=int(SAMPLE/NUM_CLASSES))
        trg_idx = mini_batch_class_balanced(target_images, sample_size=int(SAMPLE/NUM_CLASSES))
        src_train_images = X_train[src_idx]
        src_train_labels = Y_train[src_idx]
        trg_test_images = target_images[trg_idx]
        trg_test_labels = target_labels[trg_idx]
            
        encoded_src_train_images = encoder(src_train_images, training=False).numpy()
        encoded_trg_test_images = encoder(trg_test_images, training=False).numpy()
            
        encoded_src_train_images_adv = carlini_wagner_l2_small(classifier, encoded_src_train_images, clip_min=np.min(encoded_src_train_images), clip_max=np.max(encoded_src_train_images), binary_search_steps=5,  max_iterations=50, batch_size=200, learning_rate=1E-1)
        cw_acc, cw_loss = eval_accuracy_disc(encoded_src_train_images_adv, src_train_labels, classifier)
        print("Attack success:", 100 - cw_acc, cw_loss)
        print("Attack distortion:", tf.reduce_mean(tf.reduce_sum(tf.square(encoded_src_train_images_adv - encoded_src_train_images), axis = 1)).numpy())
        WDs_sadv = compute_WD(encoded_src_train_images, src_train_labels, encoded_src_train_images_adv, src_train_labels)
        WDs_target = compute_WD(encoded_src_train_images, src_train_labels, encoded_trg_test_images, trg_test_labels)
        RHO = FACTOR * WDs_sadv
        print("updating RHO:", RHO)
        print(WDs_sadv, "RHO_target:", WDs_target/WDs_sadv)