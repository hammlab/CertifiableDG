import numpy as np
import tensorflow as tf
from utils import load_rotated_mnist, eval_accuracy, mini_batch_class_balanced, eval_accuracy_cdan
from models import representationNN, classificationNN, domain_predictor_cdan
import argparse

parser = argparse.ArgumentParser(description='Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--TARGET', type=str, default="2", help='Target domain')
parser.add_argument('--METHOD', type=str, default="CDAN", help='ERM or WD')
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

domain_classifier = domain_predictor_cdan(encoder, 2, REP_DIM*NUM_CLASSES)

optimizer_encoder = tf.keras.optimizers.Adam(1E-3, beta_1=0.5)
optimizer_logits = tf.keras.optimizers.Adam(1E-3, beta_1=0.5)
optimizer_dc = tf.keras.optimizers.Adam(1E-3, beta_1=0.5)

ce_loss_none = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

ckpt = tf.train.Checkpoint(encoder = encoder, classifier = classifier)
ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=1) 

@tf.function
def train_step_da_1(main_data, main_labels, domain_data, opposite_domain_labels):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        shared_main = encoder(main_data, training=True)
        main_logits = classifier(shared_main, training=True)
        main_loss = ce_loss_none(main_labels, main_logits)
        
        combined_features  = encoder(domain_data, training=True)
        combined_logits  = classifier(combined_features, training=True)
        combined_softmax  = tf.nn.softmax(combined_logits)
        
        
        combined_features_reshaped = tf.reshape(combined_features, [-1, 1, 128]) 
        combined_softmax_reshaped = tf.reshape(combined_softmax, [-1, 10, 1])
        domain_input = tf.matmul(combined_softmax_reshaped, combined_features_reshaped) 
        domain_input_reshaped = tf.reshape(domain_input, [-1, 1280])
        domain_logits = domain_classifier(domain_input_reshaped, training=True)
        
        domain_loss = ce_loss_none(opposite_domain_labels, domain_logits)
        
        loss = tf.reduce_mean(main_loss) + tf.reduce_mean(domain_loss)
            
    gradients_shared = tape.gradient(loss, encoder.trainable_variables)
    gradients_main_classifier = tape.gradient(main_loss, classifier.trainable_variables)
    
    optimizer_encoder.apply_gradients(zip(gradients_shared, encoder.trainable_variables))
    optimizer_logits.apply_gradients(zip(gradients_main_classifier, classifier.trainable_variables))
    

@tf.function
def train_step_da_2(domain_data, true_domain_labels):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        combined_features  = encoder(domain_data, training=True)
        combined_logits  = classifier(combined_features, training=True)
        combined_softmax  = tf.nn.softmax(combined_logits)
        
        
        combined_features_reshaped = tf.reshape(combined_features, [-1, 1, 128]) 
        combined_softmax_reshaped = tf.reshape(combined_softmax, [-1, 10, 1])
        domain_input = tf.matmul(combined_softmax_reshaped, combined_features_reshaped) 
        domain_input_reshaped = tf.reshape(domain_input, [-1, 1280])
        domain_logits = domain_classifier(domain_input_reshaped, training=True)
        
        domain_loss = ce_loss_none(true_domain_labels, domain_logits)
        
        loss = tf.reduce_mean(domain_loss)
        
    gradients_domain_classifier = tape.gradient(loss, domain_classifier.trainable_variables)
    optimizer_dc.apply_gradients(zip(gradients_domain_classifier, domain_classifier.trainable_variables))


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
        
        x_combined = np.concatenate([domainwise_data[0], domainwise_data[1]])
        
        y_dc_source_init = np.zeros(len(domainwise_data[0]))
        y_dc_target_init = np.ones(len(domainwise_data[1]))
        
        y_dc_correct_combined = tf.keras.utils.to_categorical(np.concatenate([y_dc_source_init, y_dc_target_init]), 2)
        y_dc_incorrect_combined = tf.keras.utils.to_categorical(np.concatenate([1-y_dc_source_init, 1-y_dc_target_init]), 2)
        
        train_step_da_2(x_combined, y_dc_correct_combined)
        train_step_da_1(xs, ys, x_combined, y_dc_incorrect_combined)
        
            
    if epoch % 100 == 0:
        print("\nTest Domains:", TRGS, SRCS, METHOD, CHECKPOINT_PATH)
        srcs_train_accuracy, _ = eval_accuracy(X_train, Y_train, encoder, classifier)
        target_test_accuracy, _ = eval_accuracy(target_images, target_labels, encoder, classifier)
        print("ERM ec Epoch:", epoch)
        print("Sources:", srcs_train_accuracy)
        print("Target:", target_test_accuracy)
        print("Disc:", eval_accuracy_cdan(x_combined, y_dc_correct_combined, encoder, classifier, domain_classifier))
        ckpt_model_save_path = ckpt_manager.save()
        print("\n")