import numpy as np
import tensorflow as tf
from utils import load_PACS, eval_accuracy, mini_batch_class_balanced, eval_accuracy_disc, compute_WD, eval_accuracy_cdan
from models import classificationNN, domain_predictor_cdan, representationNN
import argparse
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from cleverhans.tf2.attacks.carlini_wagner_l2_small import carlini_wagner_l2 as carlini_wagner_l2_small

parser = argparse.ArgumentParser(description='Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--TARGET', type=str, default="0", help='Target domain')
parser.add_argument('--FACTOR', type=str, default="0.5", help='factor of s_adv')
args = parser.parse_args()

METHOD = "CDAN"
FACTOR = float(args.FACTOR)

SRCS = [0,1,3]
TRGS = [int(args.TARGET)]
SRCS.remove(TRGS[0])

CHECKPOINT_PATH = "./checkpoints/rep_dro_dg_"+METHOD+ "_" + str(TRGS[0]) +"_factor_"+str(FACTOR)

EPOCHS = 31
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
src_data, src_labels, _, _, target_data, target_labels = load_PACS(SRCS, TRGS)

print("Loaded")
D_list = []

for d in range(0, len(src_data)):
    D_list.append(np.ones(len(src_data[d])) * d)

X_train = [item for sublist in src_data for item in sublist]
Y_train = [item for sublist in src_labels for item in sublist]
D_train = [item for sublist in D_list for item in sublist]

X_test = [item for sublist in target_data for item in sublist]
Y_test = [item for sublist in target_labels for item in sublist]


print(np.min(X_train), np.max(X_train))
X_train = np.array(X_train, dtype=np.float32).reshape([-1, HEIGHT, WIDTH, NCH])
target_images = np.array(X_test, dtype=np.float32).reshape([-1, HEIGHT, WIDTH, NCH])

X_train = preprocess_input(X_train * 255)
target_images =  preprocess_input(target_images * 255)


Y_train = tf.keras.utils.to_categorical(Y_train, NUM_CLASSES)
target_labels = tf.keras.utils.to_categorical(Y_test, NUM_CLASSES)

D_train = tf.keras.utils.to_categorical(D_train, NUM_DOMAINS)

print(D_train.shape, X_train.shape, Y_train.shape)
print(target_images.shape, target_labels.shape)
print(np.min(X_train), np.max(X_train))

REP_DIM = 128
base_model = ResNet50(weights='imagenet', include_top=False, pooling="avg")
encoder = representationNN(2048)
classifier = classificationNN(REP_DIM, NUM_CLASSES)
domain_classifier = domain_predictor_cdan(2, REP_DIM*NUM_CLASSES)

optimizer_base_model = tf.keras.optimizers.Adam(1E-5, beta_1=0.5)
optimizer_encoder = tf.keras.optimizers.Adam(1E-5, beta_1=0.5)
optimizer_logits = tf.keras.optimizers.Adam(1E-4, beta_1=0.5)

optimizer_pretrain_base_model = tf.keras.optimizers.Adam(1E-5, beta_1=0.5)
optimizer_pretrain_encoder = tf.keras.optimizers.Adam(1E-5, beta_1=0.5)
optimizer_pretrain_logits = tf.keras.optimizers.Adam(1E-4, beta_1=0.5)

optimizer_adv = tf.keras.optimizers.Adam(1E-2, beta_1=0.5)
optimizer_dc = tf.keras.optimizers.Adam(1E-4, beta_1=0.5)

ce_loss_none = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

adv_z = tf.Variable(tf.zeros([BATCH_SIZE, REP_DIM]), trainable=True, name="adv_z")

ckpt = tf.train.Checkpoint(base_model = base_model, encoder = encoder, classifier = classifier)
ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=1) 

@tf.function
def train_step_da_1_pretrain(main_data, main_labels, domain_data, opposite_domain_labels):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        shared_main = encoder(base_model(main_data, training=False), training=True)
        main_logits = classifier(shared_main, training=True)
        main_loss = ce_loss_none(main_labels, main_logits)
        
        combined_features  = encoder(base_model(domain_data, training=False), training=True)
        combined_logits  = classifier(combined_features, training=True)
        combined_softmax  = tf.nn.softmax(combined_logits)
        
        combined_features_reshaped = tf.reshape(combined_features, [-1, 1, REP_DIM]) 
        combined_softmax_reshaped = tf.reshape(combined_softmax, [-1, NUM_CLASSES, 1])
        domain_input = tf.matmul(combined_softmax_reshaped, combined_features_reshaped) 
        domain_input_reshaped = tf.reshape(domain_input, [-1, REP_DIM*NUM_CLASSES])
        domain_logits = domain_classifier(domain_input_reshaped, training=True)
        
        domain_loss = ce_loss_none(opposite_domain_labels, domain_logits)
        
        loss = tf.reduce_mean(main_loss) + tf.reduce_mean(domain_loss)
          
    gradients_base_model = tape.gradient(loss, base_model.trainable_variables)
    gradients_shared = tape.gradient(loss, encoder.trainable_variables)
    gradients_main_classifier = tape.gradient(loss, classifier.trainable_variables)
    
    optimizer_pretrain_base_model.apply_gradients(zip(gradients_base_model, base_model.trainable_variables))
    optimizer_pretrain_encoder.apply_gradients(zip(gradients_shared, encoder.trainable_variables))
    optimizer_pretrain_logits.apply_gradients(zip(gradients_main_classifier, classifier.trainable_variables))
    

@tf.function
def train_step_da_1(main_data, encoded_adv_data, main_labels, domain_data, opposite_domain_labels):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        shared_main = encoder(base_model(main_data, training=False), training=True)
        main_logits = classifier(shared_main, training=True)
        main_loss = ce_loss_none(main_labels, main_logits)
        
        outputs_adv = classifier(encoded_adv_data, training=True)
        loss_cls_adv = ce_loss_none(main_labels, outputs_adv)
        
        combined_features  = encoder(base_model(domain_data, training=False), training=True)
        combined_logits  = classifier(combined_features, training=True)
        combined_softmax  = tf.nn.softmax(combined_logits)
        
        combined_features_reshaped = tf.reshape(combined_features, [-1, 1, REP_DIM]) 
        combined_softmax_reshaped = tf.reshape(combined_softmax, [-1, NUM_CLASSES, 1])
        domain_input = tf.matmul(combined_softmax_reshaped, combined_features_reshaped) 
        domain_input_reshaped = tf.reshape(domain_input, [-1, REP_DIM*NUM_CLASSES])
        domain_logits = domain_classifier(domain_input_reshaped, training=True)
        
        domain_loss = ce_loss_none(opposite_domain_labels, domain_logits)
        
        loss_rep = tf.reduce_mean(main_loss) + tf.reduce_mean(domain_loss)
        loss_cls = tf.reduce_mean(main_loss) + tf.reduce_mean(loss_cls_adv)
            
    gradients_base_model = tape.gradient(loss_rep, base_model.trainable_variables)
    gradients_shared = tape.gradient(loss_rep, encoder.trainable_variables)
    gradients_main_classifier = tape.gradient(loss_cls, classifier.trainable_variables)
    
    optimizer_base_model.apply_gradients(zip(gradients_base_model, base_model.trainable_variables))
    optimizer_encoder.apply_gradients(zip(gradients_shared, encoder.trainable_variables))
    optimizer_logits.apply_gradients(zip(gradients_main_classifier, classifier.trainable_variables))
    

@tf.function
def train_step_da_2(domain_data, true_domain_labels):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        combined_features  = encoder(base_model(domain_data, training=False), training=True)
        combined_logits  = classifier(combined_features, training=True)
        combined_softmax  = tf.nn.softmax(combined_logits)
        
        
        combined_features_reshaped = tf.reshape(combined_features, [-1, 1, REP_DIM]) 
        combined_softmax_reshaped = tf.reshape(combined_softmax, [-1, NUM_CLASSES, 1])
        domain_input = tf.matmul(combined_softmax_reshaped, combined_features_reshaped) 
        domain_input_reshaped = tf.reshape(domain_input, [-1, REP_DIM*NUM_CLASSES])
        domain_logits = domain_classifier(domain_input_reshaped, training=True)
        
        domain_loss = ce_loss_none(true_domain_labels, domain_logits)
        
        loss = tf.reduce_mean(domain_loss)
        
    gradients_domain_classifier = tape.gradient(loss, domain_classifier.trainable_variables)
    optimizer_dc.apply_gradients(zip(gradients_domain_classifier, domain_classifier.trainable_variables))

@tf.function
def train_step_adv_z(encoded_data_adv, encoded_data_orig, labels, GAMMA_1):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        adv_z.assign(encoded_data_adv)
        
        rep_orig = encoded_data_orig
        distortion = tf.reduce_sum(tf.square(adv_z - rep_orig), axis = 1)
        
        adv_logits = classifier(adv_z, training=True)
        
        adv_loss = ce_loss_none(labels, adv_logits)
        
        loss = -(adv_loss - tf.cast(GAMMA_1, tf.float32) * distortion)
      
    gradients_adv = tape.gradient(loss, adv_z)
    optimizer_adv.apply_gradients(zip([gradients_adv], [adv_z]))


encoded_X_adv = np.zeros([len(X_train), REP_DIM], dtype=np.float32)
for epoch in range(EPOCHS):
    print(epoch)
    nb_batches_train = int(len(X_train)/BATCH_SIZE)
    ind_shuf = np.arange(len(X_train))
    np.random.shuffle(ind_shuf)
    
    avg_distortion = 0
    for batch in range(nb_batches_train):
        ind_b = range(BATCH_SIZE * batch, min(BATCH_SIZE * (1+batch), len(X_train)))
        ind = ind_shuf[ind_b]
        
        xs = np.array(X_train[ind])
        ys = np.array(Y_train[ind])
        ds = np.array(D_train[ind])
        
        
        encoded_xs = np.array(encoder(base_model(xs, training=False), training=False).numpy())
        if epoch >= PRETRAIN_STEPS:
            
            encoded_xs_adv = np.array(encoded_xs)
            
                
            for _ in range(ADV_STEPS):
                
                train_step_adv_z(tf.convert_to_tensor(encoded_xs_adv, tf.float32), tf.convert_to_tensor(encoded_xs, tf.float32), tf.convert_to_tensor(ys, tf.float32), tf.convert_to_tensor(GAMMA, tf.float32))
                
                encoded_xs_adv = np.array(adv_z.numpy())
                
              
                distortion_batch = tf.reduce_sum(tf.square(encoded_xs - encoded_xs_adv), axis = 1)
                grad_GAMMA = RHO**2 - np.mean(distortion_batch.numpy())
                
                GAMMA = np.clip(GAMMA - 0.01 * grad_GAMMA, 1E-5, 1E3)
                
            
            
            avg_distortion += np.mean(distortion_batch.numpy())
            
        ##########
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
        
        for _ in range(1):
            train_step_da_2(x_combined, y_dc_correct_combined)
        
        if epoch >= PRETRAIN_STEPS:
            train_step_da_1(xs, encoded_xs_adv, ys, x_combined, y_dc_incorrect_combined)
            
        if epoch < PRETRAIN_STEPS:
            train_step_da_1_pretrain(xs, ys, x_combined, y_dc_incorrect_combined)
            
    if epoch % 1 == 0:
        print("\nTest Domains:", TRGS, SRCS, METHOD, CHECKPOINT_PATH)
        srcs_train_accuracy, _ = eval_accuracy(X_train, Y_train, base_model, encoder, classifier)
        target_test_accuracy, _ = eval_accuracy(target_images, target_labels, base_model, encoder, classifier)
        print("ERM ec Epoch:", epoch)
        print("Sources:", srcs_train_accuracy)
        print("Target:", target_test_accuracy)
        print("Disc:", eval_accuracy_cdan(x_combined, y_dc_correct_combined, base_model, encoder, classifier, domain_classifier))
        ckpt_model_save_path = ckpt_manager.save()
        
    if epoch >= PRETRAIN_STEPS: 
        encoded_source_adv_test_accuracy, encoded_source_adv_test_loss = eval_accuracy_disc(encoded_xs_adv, ys, classifier)
        print("RHO:", RHO, "GAMMA:", GAMMA, "GRAD GAMMA:", grad_GAMMA)
        
        print("Distortion:", avg_distortion/nb_batches_train, np.mean(distortion_batch.numpy()), np.max(distortion_batch.numpy()), np.min(distortion_batch.numpy()),  len(np.argwhere(distortion_batch.numpy()<2*RHO**2).flatten()))
        print("adv batch acc:", encoded_source_adv_test_accuracy, encoded_source_adv_test_loss)
        
        
        
    if epoch%2 == 0 or epoch == PRETRAIN_STEPS-1:    
        
        src_idx = np.random.choice(len(Y_train), 1000) 
        trg_idx = np.random.choice(len(target_labels), 1000) 
        src_train_images = np.array(X_train[src_idx])
        src_train_labels = np.array(Y_train[src_idx])
        trg_test_images = np.array(target_images[trg_idx])
        trg_test_labels = np.array(target_labels[trg_idx])
        
        print(int(SAMPLE/NUM_CLASSES), src_train_images.shape, trg_test_images.shape)
            
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
                
        
                
        encoded_src_train_images_adv = carlini_wagner_l2_small(classifier, encoded_src_train_images, clip_min=np.min(encoded_src_train_images), clip_max=np.max(encoded_src_train_images), binary_search_steps=5, 
                                                               max_iterations=500, batch_size=100, learning_rate=1E-2)
        cw_acc, cw_loss = eval_accuracy_disc(encoded_src_train_images_adv, src_train_labels, classifier)
        print("Attack success:", 100 - cw_acc, cw_loss)
        print("Attack distortion:", tf.reduce_mean(tf.reduce_sum(tf.square(encoded_src_train_images_adv - encoded_src_train_images), axis = 1)).numpy())
        WDs_sadv = compute_WD(encoded_src_train_images, src_train_labels, encoded_src_train_images_adv, src_train_labels)
        WDs_target = compute_WD(encoded_src_train_images, src_train_labels, encoded_trg_test_images, trg_test_labels)
        RHO = FACTOR * WDs_sadv
        print("updating RHO:", RHO)
        print(WDs_sadv, "Factor_target:", WDs_target/WDs_sadv)
        
        