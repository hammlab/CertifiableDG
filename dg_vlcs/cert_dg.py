import numpy as np
import tensorflow as tf
from utils import load_VLCS, eval_accuracy, mini_batch_class_balanced, eval_accuracy_disc, compute_WD, clip_eta
from models import classificationNN, representationNN
from tensorflow.keras.applications.resnet50 import ResNet50
import argparse
from tensorflow.keras.applications.resnet50 import preprocess_input
from cleverhans.tf2.attacks.carlini_wagner_l2_small import carlini_wagner_l2 as carlini_wagner_l2_small

parser = argparse.ArgumentParser(description='Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--TARGET', type=str, default="3", help='Target domain')
parser.add_argument('--METHOD', type=str, default="G2DM", help='G2DM or WD')
parser.add_argument('--FACTOR', type=str, default="1.0", help='factor of s_adv')
args = parser.parse_args()

NAME = args.METHOD
FACTOR = float(args.FACTOR)

METHOD = "vanilla_dg"
#METHOD = "rep_dro_dg"

SRCS = [0,2,3]
TRGS = [int(args.TARGET)]
SRCS.remove(TRGS[0])
print(SRCS, TRGS)

CHECKPOINT_PATH = "./checkpoints/" + METHOD  + "_" + NAME + "_" + str(3)

if "dro" in METHOD:
    CHECKPOINT_PATH = CHECKPOINT_PATH + "_factor_" + str(args.FACTOR)


NUM_CLASSES = 5
NUM_DOMAINS = len(SRCS)

HEIGHT = 224
WIDTH = 224
NCH = 3

REP_DIM = 128

BATCH_SIZE = 100
EPOCHS = 10000
ADV_STEPS = 1

SAMPLE = 1000

# Load Dataset
print("Loading data")
_,_, src_test_data, src_test_labels, target_test_data, target_test_labels = load_VLCS(SRCS, TRGS)

print("Loaded")
D_list = []

for d in range(0, len(src_test_data)):
    D_list.append(np.ones(len(src_test_data[d])) * d)

X_train = [item for sublist in src_test_data for item in sublist]
Y_train = [item for sublist in src_test_labels for item in sublist]
D_train = [item for sublist in D_list for item in sublist]

X_train = np.array(X_train, dtype=np.float32).reshape([-1, HEIGHT, WIDTH, NCH])[:400]
Y_train = tf.keras.utils.to_categorical(Y_train, NUM_CLASSES)[:400]
X_train = preprocess_input(X_train * 255)

X_test = [item for sublist in target_test_data for item in sublist]
Y_test = [item for sublist in target_test_labels for item in sublist]

target_images = preprocess_input(255 * np.array(X_test, dtype=np.float32).reshape([-1, HEIGHT, WIDTH, NCH]))
target_labels = tf.keras.utils.to_categorical(Y_test, NUM_CLASSES)

D_train = tf.keras.utils.to_categorical(D_train, NUM_DOMAINS)

base_model = ResNet50(weights='imagenet', include_top=False, pooling="avg")
encoder = representationNN(2048)
classifier = classificationNN(REP_DIM, NUM_CLASSES)

adv_z = tf.Variable(tf.zeros([BATCH_SIZE, REP_DIM]), trainable=True, name="adv_z")

optimizer_adv = tf.keras.optimizers.Adam(1E-2, beta_1=0.5)

ce_loss_none = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

ckpt = tf.train.Checkpoint(base_model = base_model, encoder = encoder, classifier = classifier)
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

################################        
source_test_images = X_train
source_test_labels = Y_train
source_test_images_adv = np.array(source_test_images, dtype=np.float32)

trg_test_images = target_images
trg_test_labels = target_labels

nb_batches_train = int(len(X_train)/BATCH_SIZE)
if len(X_train)%BATCH_SIZE!=0:
    nb_batches_train+=1
for batch in range(nb_batches_train):
    ind_batch = range(BATCH_SIZE * batch, min(BATCH_SIZE * (1+batch), len(X_train)))
    if batch == 0:
        encoded_source_test_images = encoder(base_model(source_test_images[ind_batch], training=False), training=False).numpy()
    else:
        encoded_source_test_images = np.concatenate([encoded_source_test_images, encoder(base_model(source_test_images[ind_batch], training=False), training=False).numpy()])

nb_batches_target = int(len(trg_test_images)/BATCH_SIZE)
if len(trg_test_images)%BATCH_SIZE!=0:
    nb_batches_target+=1
for batch in range(nb_batches_target):
    ind_batch = range(BATCH_SIZE * batch, min(BATCH_SIZE * (1+batch), len(trg_test_images)))
    if batch == 0:
        encoded_target_test_images = encoder(base_model(trg_test_images[ind_batch], training=False), training=False).numpy()
    else:
        encoded_target_test_images = np.concatenate([encoded_target_test_images, encoder(base_model(trg_test_images[ind_batch], training=False), training=False).numpy()])
        

####################################################
print("Computing S_adv")
encoded_source_test_images_adv_cw = carlini_wagner_l2_small(classifier, encoded_source_test_images, clip_min=np.min(encoded_source_test_images), 
clip_max=np.max(encoded_source_test_images), binary_search_steps=5,  max_iterations=1000, batch_size=100, learning_rate=1E-1)

cw_acc, cw_loss = eval_accuracy_disc(encoded_source_test_images_adv_cw, source_test_labels, classifier)
print("Attack success:", 100 - cw_acc, cw_loss)
print("Attack distortion:", tf.reduce_mean(tf.reduce_sum(tf.square(encoded_source_test_images_adv_cw - encoded_source_test_images), axis = 1)).numpy())

WD_Sadv = compute_WD(encoded_source_test_images, source_test_labels, encoded_source_test_images_adv_cw, source_test_labels, NUM_CLASSES=NUM_CLASSES)
print("WD(S_test_adv, S_test):", WD_Sadv)


#####################################################
RHOS = np.arange(0.0, 2.1, 0.2) * WD_Sadv
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
                
                xt = encoded_source_test_images[ind]
                yt = source_test_labels[ind]
        
                xt_adv = encoded_source_test_images_adv_z[ind]
                
                for _ in range(ADV_STEPS):
                    train_step_adv_z(tf.convert_to_tensor(xt_adv, tf.float32), tf.convert_to_tensor(xt, tf.float32), tf.convert_to_tensor(yt, tf.float32), tf.convert_to_tensor(GAMMA, tf.float32))
                    
                    xt_adv = adv_z.numpy()
                    encoded_source_test_images_adv_z[ind] = xt_adv

                    distortion_batch = tf.reduce_sum(tf.square(xt - xt_adv), axis = 1)

                    
            
            distortion_all = tf.reduce_sum(tf.square(encoded_source_test_images - encoded_source_test_images_adv_z), axis = 1)
            grad_GAMMA = RHO**2 - np.mean(distortion_all.numpy())
            
            GAMMA = tf.clip_by_value(GAMMA - 0.01 * grad_GAMMA, 1E-5, 1E3).numpy()
            
            
            if epoch > 20 and abs(tf.reduce_mean(distortion_all).numpy() - RHO**2) < 1E-4: 
                print("breaking:", abs(np.mean(distortion_all.numpy()) - RHO**2))
                break
            
            if epoch % 1000 == 0:
                
                encoded_source_adv_test_accuracy, encoded_source_adv_test_loss = eval_accuracy_disc(encoded_source_test_images_adv_z, source_test_labels, classifier)
                print(CHECKPOINT_PATH)
                print(RHO/WD_Sadv, "RHO:", RHO, "GAMMA:", GAMMA, "GRAD GAMMA:", grad_GAMMA)
                print("Distortion:", np.mean(distortion_all.numpy()), np.min(distortion_all.numpy()), np.max(distortion_all.numpy()), len(np.argwhere(distortion_all.numpy()<RHO**2).flatten()))
                print(tf.reduce_mean(distortion_all).numpy() - RHO**2, np.mean(distortion_batch.numpy()))
                print(encoded_source_adv_test_accuracy, encoded_source_adv_test_loss, "\n")
        
    encoded_source_adv_test_accuracy, encoded_source_adv_test_loss = eval_accuracy_disc(encoded_source_test_images_adv_z, source_test_labels, classifier)
    DRO_loss = GAMMA * RHO**2 + encoded_source_adv_test_loss - np.mean(GAMMA * distortion_all.numpy())
    Loss_at_RHO.append(np.round(DRO_loss, 4))
    print("DRO loss:", DRO_loss, "\n\n")
    
print(CHECKPOINT_PATH)
print(Loss_at_RHO)  
