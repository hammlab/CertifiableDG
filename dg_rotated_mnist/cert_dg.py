import numpy as np
import tensorflow as tf
from utils import load_rotated_mnist, eval_accuracy,  eval_accuracy_disc, compute_WD, mini_batch_class_balanced
from models import representationNN, classificationNN, representationNN_small
from cleverhans.tf2.attacks.carlini_wagner_l2_small import carlini_wagner_l2 as carlini_wagner_l2_small
import argparse

parser = argparse.ArgumentParser(description='Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--TARGET', type=str, default="2", help='Target domain')
parser.add_argument('--NAME', type=str, default="DANN", help='DANN, WD, VREX, CDAN')
parser.add_argument('--FACTOR', type=str, default="0.5", help='factor of s_adv')
args = parser.parse_args()

NAME = args.NAME
FACTOR = float(args.FACTOR)

METHOD = "vanilla_dg"
#METHOD = "rep_dro_dg"

CHECKPOINT_PATH = "./checkpoints/" + METHOD + "_" + NAME

if "dro" in METHOD:
    CHECKPOINT_PATH = CHECKPOINT_PATH + "_factor_" + str(args.FACTOR)

SRCS = [0,1,2]
TRGS = [int(args.TARGET)]
SRCS.remove(TRGS[0])

NUM_CLASSES = 10
NUM_DOMAINS = len(SRCS)

HEIGHT = 28
WIDTH = 28
NCH = 1

REP_DIM = 128
SAMPLE = 1000
LAMBDA = 1

BATCH_SIZE = 200
EPOCHS = 501
ADV_STEPS = 1


# Load Dataset
print("Loading data")
_, _, src_test_data, src_test_labels, _, _, target_test_data, target_test_labels = load_rotated_mnist(SRCS, TRGS)

print("Loaded")
D_list = []

for d in range(0, len(src_test_data)):
    D_list.append(np.ones(len(src_test_data[d])) * d)

X_train = [item for sublist in src_test_data for item in sublist]
Y_train = [item for sublist in src_test_labels for item in sublist]
D_train = [item for sublist in D_list for item in sublist]

X_train = np.array(X_train, dtype=np.float32).reshape([-1, HEIGHT, WIDTH, NCH])
Y_train = tf.keras.utils.to_categorical(Y_train, NUM_CLASSES)

target_images = [np.array(target_test_data[i], dtype=np.float32).reshape([-1, HEIGHT, WIDTH, NCH]) for i in range(len(TRGS))]
target_labels = [tf.keras.utils.to_categorical(target_test_labels[i], NUM_CLASSES) for i in range(len(TRGS))]

D_train = tf.keras.utils.to_categorical(D_train, NUM_DOMAINS)

encoder = representationNN(X_train.shape)
classifier = classificationNN(REP_DIM, NUM_CLASSES)

adv_z = tf.Variable(tf.zeros([BATCH_SIZE, REP_DIM]), trainable=True, name="adv_z")

full_outputs = classifier(encoder.output)
full_models = tf.keras.Model(inputs=encoder.inputs, outputs=full_outputs)

optimizer_adv = tf.keras.optimizers.Adam(1E-2, beta_1=0.5)


ce_loss_none = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

ckpt = tf.train.Checkpoint(encoder = encoder, classifier = classifier)
ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=1)
ckpt.restore(ckpt_manager.latest_checkpoint)
    
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
    return gradients_adv, loss
    
################################        
  
src_idx = mini_batch_class_balanced(Y_train, sample_size=int(SAMPLE/NUM_CLASSES))
trg_idx = [mini_batch_class_balanced(target_labels[i], sample_size=int(SAMPLE/NUM_CLASSES)) for i in range(len(TRGS))]

source_test_images = X_train[src_idx]
source_test_labels = Y_train[src_idx]
source_test_images_adv = np.array(source_test_images, dtype=np.float32)

trg_test_images = [target_images[i][trg_idx[i]] for i in range(len(TRGS))]
trg_test_labels = [target_labels[i][trg_idx[i]] for i in range(len(TRGS))]

encoded_source_test_images = encoder(source_test_images, training=False).numpy()
encoded_target_test_images = [encoder(trg_test_images[i], training=False).numpy()  for i in range(len(TRGS))]

####################################################
print("Computing S_adv")
encoded_source_test_images_adv_cw = carlini_wagner_l2_small(classifier, encoded_source_test_images, clip_min=np.min(encoded_source_test_images), clip_max=np.max(encoded_source_test_images), binary_search_steps=5,  max_iterations=1000, batch_size=500, learning_rate=1E-2)
cw_acc, cw_loss = eval_accuracy_disc(encoded_source_test_images_adv_cw, source_test_labels, classifier)
print("Attack success:", 100 - cw_acc, cw_loss)
print("Attack distortion:", tf.reduce_mean(tf.reduce_sum(tf.square(encoded_source_test_images_adv_cw - encoded_source_test_images), axis = 1)).numpy())

WD_Sadv = compute_WD(encoded_source_test_images, source_test_labels, encoded_source_test_images_adv_cw, source_test_labels)
print("WD(S_test_adv, S_test):", WD_Sadv)

#####################################################
RHOS = np.arange(0, 2.2, 0.2) * WD_Sadv

Accuracy_at_RHO = []
Loss_at_RHO = []
for RHO in RHOS:
    encoded_source_test_images_adv_z = np.array(encoded_source_test_images)
    GAMMA = np.float32(1.)
    distortion_all = tf.zeros([1])
    if RHO > 0 :
        
        for epoch in range(EPOCHS):        
            
            nb_batches_test = int(len(source_test_images)/(BATCH_SIZE))
            ind_shuf = np.arange(len(source_test_images))
            np.random.shuffle(ind_shuf)

            for batch in range(nb_batches_test):
                ind_b = range(BATCH_SIZE * batch, min(BATCH_SIZE * (1+batch), len(source_test_images)))
                ind = ind_shuf[ind_b]
                
                xt = np.array(encoded_source_test_images[ind])
                yt = source_test_labels[ind]
        
                xt_adv = encoded_source_test_images_adv_z[ind]
                
                for _ in range(ADV_STEPS):
                    grad_z, loss_batch = train_step_adv_z(tf.convert_to_tensor(xt_adv, tf.float32), tf.convert_to_tensor(xt, tf.float32), tf.convert_to_tensor(yt, tf.float32), tf.convert_to_tensor(GAMMA, tf.float32))                  
                    xt_adv = adv_z.numpy()
                    
                    encoded_source_test_images_adv_z[ind] = xt_adv
                    
                    distortion_batch = tf.reduce_sum(tf.square(xt - xt_adv), axis = 1)
                    _, adv_loss_batch = eval_accuracy_disc(encoded_source_test_images_adv_z, source_test_labels, classifier)
                    
            
            distortion_all = tf.reduce_sum(tf.square(encoded_source_test_images - encoded_source_test_images_adv_z), axis = 1)
            grad_GAMMA_all = RHO**2 - np.mean(distortion_all.numpy())
            
            adv_logits_all = classifier(encoded_source_test_images_adv_z, training=False)
            adv_loss_all = ce_loss_none(source_test_labels, adv_logits_all)
            
            mean_phi_all = np.mean(adv_loss_all.numpy() - GAMMA * distortion_all.numpy())
            
            DRO_loss = GAMMA * RHO**2 + mean_phi_all
            
                           
            if NAME == 'WM':
                GAMMA = tf.clip_by_value(GAMMA - 0.1 * grad_GAMMA_all, 1E-10, 1E3).numpy()
            else:
                GAMMA = tf.clip_by_value(GAMMA - 0.01 * grad_GAMMA_all, 1E-10, 1E3).numpy()
            
    if RHO == 0 :
        DRO_loss = eval_accuracy_disc(encoded_source_test_images_adv_z, source_test_labels, classifier)[1]
    Loss_at_RHO.append(np.round(DRO_loss, 4))
    print("\n\n")
    
print(CHECKPOINT_PATH)
print(Loss_at_RHO)   
