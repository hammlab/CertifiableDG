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
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

ce_loss_none = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

def compute_WD(data_S, labels_S, data_T, labels_T, NUM_CLASSES=7, HEIGHT=28, WIDTH=28, NCH=1):
    
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

def load_PACS(sources, targets):
    
    root = "path/DomainBed/domainbed/data/PACS"

    environments = [f.name for f in os.scandir(root) if f.is_dir()]
    environments = sorted(environments)
    #['art_painting', 'cartoon', 'photo', 'sketch']
    
    common_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        
    ])
    
    
    
    src_data = []
    src_labels = []
    src_test_data = []
    src_test_labels = []
    
    target_data = []
    target_labels = []
   

    for i, environment in enumerate(environments):
    
        if i in sources or i in targets:
            
            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path, transform=common_transform)
            
        
            loader = DataLoader(env_dataset, len(env_dataset))
            dataset_array = next(iter(loader))[0].permute(0, 2, 3, 1).numpy()
            dataset_labels_array = next(iter(loader))[1].numpy()
            
            
            if i in sources:
                
                for label in range(len(np.unique(dataset_labels_array))):
                    label_indices = np.argwhere(dataset_labels_array == label).flatten()
                    if label == 0:
                        train_indices = label_indices[:int(0.9 * len(label_indices))]
                        test_indices = label_indices[int(0.9 * len(label_indices)):]
                    else:
                        train_indices = np.concatenate([train_indices, label_indices[:int(0.9 * len(label_indices))]])
                        test_indices = np.concatenate([test_indices, label_indices[int(0.9 * len(label_indices)):]])
                        
                
                idxes = np.arange(len(train_indices))
                np.random.shuffle(idxes)
                train_indices = train_indices[idxes]
                
                
                
                src_data.append(dataset_array[train_indices])
                src_labels.append(dataset_labels_array[train_indices])
                src_test_data.append(dataset_array[test_indices])
                src_test_labels.append(dataset_labels_array[test_indices])
            
            elif i in targets:
                target_data.append(dataset_array)
                target_labels.append(dataset_labels_array)
                
        

    return src_data, src_labels, src_test_data, src_test_labels, target_data, target_labels

def eval_accuracy(x_test, y_test, base_model, encoder, classifier):
    correct = 0
    points = 0
    loss = 0
    batch_size = 50
    nb_batches = int(len(x_test)/batch_size)
    if len(x_test)%batch_size!= 0:
        nb_batches += 1
    
    for batch in range(nb_batches):
        ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(x_test)))
        rep = encoder(base_model(x_test[ind_batch], training=False), training=False)
        pred = classifier(rep, training=False)
        
        correct += np.sum(np.argmax(pred,1) == np.argmax(y_test[ind_batch],1))
        points += len(ind_batch)
        loss += np.sum(ce_loss_none(y_test[ind_batch], pred).numpy())
    
    return (correct / np.float32(points))*100., loss/ np.float32(points)

def eval_accuracy_cdan(x_test, y_test, base_model, encoder, classifier, domain_classifier, loss_fn = 'ce'):
    correct = 0
    points = 0
    loss = 0
    batch_size = 200
    nb_batches = int(len(x_test)/batch_size)
    if len(x_test)%batch_size!= 0:
        nb_batches += 1
    
    for batch in range(nb_batches):
        ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(x_test)))
        rep = encoder(base_model(x_test[ind_batch], training=False), training=False)
        
        combined_logits  = classifier(rep, training=True)
        combined_softmax  = tf.nn.softmax(combined_logits)
        
        combined_features_reshaped = tf.reshape(rep, [-1, 1, 128]) 
        combined_softmax_reshaped = tf.reshape(combined_softmax, [-1, 7, 1])
        domain_input = tf.matmul(combined_softmax_reshaped, combined_features_reshaped) 
        domain_input_reshaped = tf.reshape(domain_input, [-1, 128*7])
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




def eval_accuracy_disc(x_test, y_test, classifier):
    correct = 0
    points = 0
    loss = 0
    batch_size = 50
    nb_batches = int(len(x_test)/batch_size)
    if len(x_test)%batch_size!= 0:
        nb_batches += 1
    
    for batch in range(nb_batches):
        ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(x_test)))
        #print(len(ind_batch))
        pred = classifier(x_test[ind_batch], training=False)
        
        correct += np.sum(np.argmax(pred,1) == np.argmax(y_test[ind_batch],1))
        points += len(ind_batch)
        loss += np.sum(ce_loss_none(y_test[ind_batch], pred).numpy())
    
    return (correct / np.float32(points))*100., loss/ np.float32(points)



def mini_batch_class_balanced(label, sample_size=20):
    ''' sample the mini-batch with class balanced
    '''
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

