import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.patches as mpatches
import numpy as np
import tensorflow as tf
import ot
from scipy.spatial.distance import cdist 

ce_loss_none = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

def compute_WD(data_S, labels_S, data_T, labels_T, NUM_CLASSES=10, HEIGHT=28, WIDTH=28, NCH=1):
    
    wd_num = 0
    wd_deno = 0
    for k in range(NUM_CLASSES):
        idx_S = np.argwhere(np.argmax(labels_S, 1) == k).flatten()
        idx_T = np.argwhere(np.argmax(labels_T, 1) == k).flatten()
        
        if len(data_S.shape) > 2:
            C = cdist(data_S[idx_S].reshape([-1, HEIGHT*WIDTH*NCH]), data_T[idx_T].reshape([-1, HEIGHT*WIDTH*NCH]), metric='sqeuclidean')
        else:
            C = cdist(data_S[idx_S], data_T[idx_T], metric='sqeuclidean')
            
        gamma = ot.emd(ot.unif(len(idx_S)), ot.unif(len(idx_T)), C)
        wd_num += np.sum(gamma * C)
        wd_deno += np.sum(gamma)
    
    WDs = np.sqrt(wd_num/wd_deno)
    
    return WDs

def load_rotated_mnist(sources, targets):
    data = np.load("../preprocess_data/data/rotated_mnist_data.npy").tolist()
    labels = np.load("../preprocess_data/data/rotated_mnist_labels.npy").tolist()
    
    test_data = np.load("../preprocess_data/data/rotated_mnist_test_data.npy").tolist()
    test_labels = np.load("../preprocess_data/data/rotated_mnist_test_labels.npy").tolist()

    src_data = []
    src_labels = []
    src_test_data = []
    src_test_labels = []
    for i in sources:
        src_data.append(data[i])
        src_labels.append(labels[i])
        src_test_data.append(test_data[i])
        src_test_labels.append(test_labels[i])
    
    target_data = []
    target_labels = []
    target_test_data = []
    target_test_labels = []
    for i in targets:
        target_data.append(data[i])
        target_labels.append(labels[i])
        target_test_data.append(test_data[i])
        target_test_labels.append(test_labels[i])

    return src_data, src_labels, src_test_data, src_test_labels, target_data, target_labels, target_test_data, target_test_labels

def eval_accuracy(x_test, y_test, encoder, classifier, loss_fn = 'ce'):
    correct = 0
    points = 0
    loss = 0
    batch_size = 200
    nb_batches = int(len(x_test)/batch_size)
    if len(x_test)%batch_size!= 0:
        nb_batches += 1
    
    for batch in range(nb_batches):
        ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(x_test)))
        rep = encoder(x_test[ind_batch], training=False)
        pred = classifier(rep, training=False)
        
        correct += np.sum(np.argmax(pred,1) == np.argmax(y_test[ind_batch],1))
        points += len(ind_batch)
        if loss_fn == 'ce':
            loss += np.sum(ce_loss_none(y_test[ind_batch], pred).numpy())
        else:
            pred_softmax = tf.nn.softmax(pred)
            
            real = tf.reduce_sum(y_test[ind_batch] * pred_softmax, 1)
            other = tf.reduce_max((1 - y_test[ind_batch]) * pred_softmax - (y_test[ind_batch] * 10000),1)
            loss += np.sum(tf.maximum(0.0, other - real + 0.1) - 0.1 * tf.maximum(0.0, -0.1 - (other - real)).numpy())
    
    return (correct / np.float32(points))*100., loss/ np.float32(points)

def eval_accuracy_cdan(x_test, y_test, encoder, classifier, domain_classifier, loss_fn = 'ce'):
    correct = 0
    points = 0
    loss = 0
    batch_size = 200
    nb_batches = int(len(x_test)/batch_size)
    if len(x_test)%batch_size!= 0:
        nb_batches += 1
    
    for batch in range(nb_batches):
        ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(x_test)))
        rep = encoder(x_test[ind_batch], training=False)
        
        combined_logits  = classifier(rep, training=True)
        combined_softmax  = tf.nn.softmax(combined_logits)
        
        combined_features_reshaped = tf.reshape(rep, [-1, 1, 128]) 
        combined_softmax_reshaped = tf.reshape(combined_softmax, [-1, 10, 1])
        domain_input = tf.matmul(combined_softmax_reshaped, combined_features_reshaped) 
        domain_input_reshaped = tf.reshape(domain_input, [-1, 1280])
        pred = domain_classifier(domain_input_reshaped, training=True)
        
        correct += np.sum(np.argmax(pred,1) == np.argmax(y_test[ind_batch],1))
        points += len(ind_batch)
        if loss_fn == 'ce':
            loss += np.sum(ce_loss_none(y_test[ind_batch], pred).numpy())
        else:
            pred_softmax = tf.nn.softmax(pred)
            
            real = tf.reduce_sum(y_test[ind_batch] * pred_softmax, 1)
            other = tf.reduce_max((1 - y_test[ind_batch]) * pred_softmax - (y_test[ind_batch] * 10000),1)
            loss += np.sum(tf.maximum(0.0, other - real + 0.1) - 0.1 * tf.maximum(0.0, -0.1 - (other - real)).numpy())
    
    return (correct / np.float32(points))*100., loss/ np.float32(points)


def eval_accuracy_disc(x_test, y_test, classifier, loss_fn = 'ce'):
    correct = 0
    points = 0
    loss = 0
    batch_size = 200
    nb_batches = int(len(x_test)/batch_size)
    if len(x_test)%batch_size!= 0:
        nb_batches += 1
    
    for batch in range(nb_batches):
        ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(x_test)))
        pred = classifier(x_test[ind_batch], training=False)
        
        correct += np.sum(np.argmax(pred,1) == np.argmax(y_test[ind_batch],1))
        points += len(ind_batch)
        if loss_fn == 'ce':
            loss += np.sum(ce_loss_none(y_test[ind_batch], pred).numpy())
        else:
            #pred_softmax = tf.nn.softmax(pred)
            
            real = tf.reduce_sum(y_test[ind_batch] * pred, 1)
            other = tf.reduce_max((1 - y_test[ind_batch]) * pred - (y_test[ind_batch] * 10000),1)
            loss += np.sum(tf.maximum(0.0, other - real + 0.1) - 0.1 * tf.maximum(0.0, -0.1 - (other - real)).numpy())
    
    return (correct / np.float32(points))*100., loss/ np.float32(points)


def mini_batch_class_balanced(label, sample_size=20):
    label = np.argmax(label, axis=1)

    n_class = len(np.unique(label))
    index = []
    for i in range(n_class):
        s_index = np.argwhere(label==i).flatten()
        np.random.shuffle(s_index)
        index.append(s_index[:sample_size])

    index = [item for sublist in index for item in sublist]
    index = np.array(index, dtype=int)
    return index

