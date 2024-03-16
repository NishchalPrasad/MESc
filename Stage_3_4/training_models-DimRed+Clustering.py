
#import sys
# caution: path[0] is reserved for script path (or '' in REPL)
#sys.path.append('MESc/Stage_3_4')

import models
import umap
from models.stacked_encoder.predictive_models import StackedEncoder, Classification_with_Clustering, Classification_without_clustering_model_, LearningRateScheduler, Classification_with_clustering_model_
from models.stacked_encoder.keras_nlp_transformer_encoder import TransformerEncoder
from metrics_calculator import absoluteFilePaths, predict_and_plot, predict_plot
import data_generators
#from models.stacked_encoder.transformer_encoder_tf import 
import tensorflow as tf
import pandas as pd
import numpy as np
import numpy as np
import keras
import os
import datetime
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, TensorBoard
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.constraints import MaxNorm
#import tensorflow_hub as hub
import numpy as np
from numpy import load
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import glob
#Training with TransformerEncoder from keras_nlp
#import keras_nlp
import argparse
import math

np.random.seed(1337)# setting the random seed value
tf.config.list_physical_devices('GPU') 
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def args_parser():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--dataset_subset", 
        type=str, 
        default=None, 
        help="Dataset to train on")
    
    args.add_argument(
        "--combine_layers", 
        type=bool, 
        default=True, 
        help="Is the extracted data combined to train?")
    
    args.add_argument(
        "--add_layers",
        type=bool,
        default=False,
        help="Is the extracted data from last four layers added together?")
    
    args.add_argument(
        "--layers_from_end",
        type=int,
        default=4,
        help="How many layers from the end are combined or added?")
    
    args.add_argument(
        "--save_model_trained_here",
        type=bool,
        default=False,
        help="Save the model trained here?")
    
    args.add_argument(
        "--self_ft_extracted",
        type=bool,
        default=True,
        help="Is the extracted data extracted from model finetuned by you?")
    
    args.add_argument(
        "--ft_model_used_for_extraction",
        type=str,
        default="ft_model",
        help="If so then path of the finetuned model which was used for the extraction")
    
    args.add_argument(
        "--pretrained_model",
        type=str,
        default="Pretrained_model",
        help="which pretrained model was finetuned (path from huggingface)?")
    args.add_argument(
        "--finetuned_on_input_len",
        type = int,
        default=512,
        help="length of inputs on which the LLM is finetuned."
    )

    #------------------for training------------------
    args.add_argument(
        "--to_test",
        type=bool,
        default=False,
        help="If this is true only then it tests on the test set")
    
    args.add_argument(
        "--to_train",
        type=bool,
        default=False,
        help="If this is true only then it trains")
    args.add_argument(
        "--minimum_number_of_chunks", 
        type=int, 
        default=None, 
        help="Documents with minimum number of 'n' chunk to consider for testing."
    )
    args.add_argument(
        "--start_chunk", 
        type=int, 
        default=None, 
        help="Starting chunk to consider for testing. Only the 'minimum_number_of_chunks' > 'start_chunk', for a document will be used for testing."
    )
    args.add_argument(
        "--eq_no_of_chunks_from_ft", 
        type=int, 
        default=None, 
        help="Equivalent number of chunks evaluated on the finetuned standalone model."
    )
    args.add_argument(
        "--eq_chunk_len", 
        type=int, 
        default=None, 
        help="Chunk-length for the finetuned standalone model used for evaluation."
    )
    args.add_argument(
        "--overlaps", 
        type=int, 
        default=100, 
        help="Overlaps while extracting the embeddings from the finetuned model and evaluating on the standalone finetuned model."
    )
    args.add_argument(
        "--current_chunk_len", 
        type=int, 
        default=512, 
        help="Length of chunks currently used for testing."
    )
    args.add_argument(
        "--train_run_number",
        type = int,
        default = 1,
        help="The run number of training i.e. which run?")
    args.add_argument(
        "--verbose",
        type=int,
        default=2,
        help="1 to show training, 2 for silent training, 3 for no display")
    
    args.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="epochs to train the model")
    
    args.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch_size")
    
    args.add_argument(
        "--load_to_retrain",
        type=bool,
        default=False,
        help="load model for retraining")
    
    args.add_argument(
        "--model_load_path",
        type=str,
        default="",
        help="load model for retraining, or testing when to_test is true and to_train is false")
    
    args.add_argument(
        "--to_freeze",
        type=bool,
        default=False,
        help="to freeze the encoder model")
    
    args.add_argument(
        "--layers_to_freeze",
        type=list,
        default=[1],
        help="Layers to freeze in the model")
    
    #------------only for encoder model
    args.add_argument(
        "--only_RNN", 
        type=bool, 
        default=False, 
        help="If use only RNN")
    args.add_argument(
        "--two_RNN", 
        type=bool, 
        default=False, 
        help="If use two RNNs")
    
    args.add_argument(
        "--max_positional_encoding",
        type=int,
        default=40,
        help="max_positional_encoding")
    
    args.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="num_layers")
    args.add_argument(
        "--positional_encoding",
        type=bool,
        default=False,
        help="To inlcude positional encoding?")
    
    args.add_argument(
        "--bilstm_before_encoder",
        type=bool,
        default=False,
        help="bilstm_before_encoder")
    args.add_argument(
        "--two_rnn_to_enc",
        type=bool,
        default=False,
        help="Two RNNs to Encoder")    
    
    args.add_argument(
        "--dropout_before_first_fnn",
        type=bool,
        default=True,
        help="dropout_before_first_fnn")
    
    args.add_argument(
        "--dropout_before_inner_fnn",
        type=bool,
        default=True,
        help="dropout_before_inner_fnn")
    
    args.add_argument(
        "--ffn_dropout_value",
        type=float,
        default=0.15,
        help="ffn_dropout_value")
    
    args.add_argument(
        "--dropout_after_first_encoder",
        type=bool,
        default=True,
        help="dropout_after_first_encoder")
    
    args.add_argument(
        "--dropout_after_second_encoder",
        type=bool,
        default=True,
        help="dropout_after_second_encoder")
    
    args.add_argument(
        "--encoder_dropout_value",
        type=float,
        default=0.25,
        help="encoder_dropout_value")
    
    args.add_argument(
        "--dff",
        type=int,
        default=2048,
        help="dff for transformer encoder block. Make this same as the hidden dimension of the transformer model (GPT-Neo, GPT-J etc.) used to extract the CLS embeddings")
    
    args.add_argument(
        "--d_model",
        type=int,
        default=768,
        help="""d_model for transformer encoder block. It is the embedding dimension for the input. 
        So for GPT-Neo and GPT-J it is 4096 i.e the the last feature dimension of the output layer of the model.""")
    
    args.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="num_heads for transformer encoder block")
    
    
    args.add_argument(
        "--include_rnn",
        type=bool,
        default=False,
        help="include RNNs for processing output after encoder models")
    
    args.add_argument(
        "--from_encoder",
        type=bool,
        default=False,
        help="connect directly to encoder or to inputs")
    
    
    #---------------------------------
    #-----------for clustering and dimensionality reduction techniques-------------------
    args.add_argument(
        "--with_clustering",
        type=bool,
        default=False,
        help="for all strategy of prediction with clustering")
    
    args.add_argument(
        "--clustering_strategy",
        type=str,
        default='hard',
        help="hard or soft clustering") 
    
    args.add_argument(
        "--dimReduction",
        type=str,
        default='pumap',
        help="pumap or kmeans") 
    
    args.add_argument(
        "--dim_reduction_metric",
        type=str,
        default='cosine',
        help="cosine or euclidean") 
    
    args.add_argument(
        "--pad_len",
        type=int,
        default=32,
        help="""The max length for the cluster labels for using with Dense layers processing. 
        Since input feature dimension for a dense layer is of fixed while model creation. 
        This decides the input feature dimension for the dense layer for clustered labels processing. 
        32 for ildc, 64 for scotus and eurlex, 128 for etchr_a, etchr_b""")
    
    args.add_argument(
        "--strategy",
        type=int,
        default=1,
        help="""
        strategy = 1.
                    Chunked CLS
                    embeddings  ------------------------> Encoder Model -> output
                        |                                   |
                Dimensionality Reduction --> Clustering ____|
                   (PrametricUMAP)
                
        strategy = 2.
                    Chunked CLS
                    embeddings           ---------------> Encoder Model -> output
                        |                |                  |
                Dimensionality Reduction --> Clustering ____|
                   (PrametricUMAP)
                   """
                   )
    
    args.add_argument(
        "--only_clustered",
        type=bool,
        default=False,
        help="only for strategy 3")
    
    args.add_argument(
        "--train_dimR",
        type=bool,
        default=False,
        help="to train dimensionality reduction model")
    
    args.add_argument(
        "--save_dimR",
        type=bool,
        default=False,
        help="to save dimensionality reduction model")
    
    args.add_argument(
        "--train_clusterer",
        type=bool,
        default=False,
        help="to train clustering model")
    
    args.add_argument(
        "--save_clusterer",
        type=bool,
        default=False,
        help="to save clustering model")

    args.add_argument(
        "--parent_dir",
        type=str,
        default="LEGAL-PE/Level-3_of_Framework/savedModels/512_input_len/",
        help="parent directory of the model (used to save the model)")
    args.add_argument(
        "--data_path",
        type=str,
        default="",
        help="Path to the training data")
    args.add_argument(
        "--edit_for_ildc",
        type=bool,
        default=False,
        help="checking for ildc")
    args = args.parse_args()

    if args.to_train == False:
        args.train_dimR = False 
        args.train_clusterer = False
    if args.train_dimR == True:
        args.save_dimR = True 
    if args.train_clusterer == True:
        args.save_clusterer = True
    if args.to_test and args.to_train==False:
        if args.model_load_path == "":
            raise ValueError("Please provide the model path to load for testing!")
    if args.combine_layers == False:
        args.layers_from_end = 1
        args.combine_layers = True
    if args.data_path == "":
        raise ValueError("Please provide the data path!")
    if args.only_RNN:
        if args.two_RNN:
            args.two_rnn_to_enc = True
    return args
    


#get files to import
def search_files(path,keyword,extension = '*.npy'):
    path_list = []
    path_list = glob.glob(path + '/' + keyword + '*.npy')
    return path_list

"""
@tf.function
def train_step(model, x, y, optimizer, loss_fn, metrics):
    with tf.GradientTape() as tape:
        logits = model(x, training = True)
        loss_val = loss_fn(y, logits)
    gradients = tape.gradient(loss_val, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients,model.trainable_weights))
    for met in metrics:
        met.update_state(y_batch_train, logits)
    return 
"""
def get_only_large_documents(
    number_of_chunks:int=None,
    indexes:int=None,
    embeddings=None,
    labels=None,
    eq_no_of_chunks_from_ft:int=None,
    eq_chunk_len:int=None,
    overlaps:int=100,
    current_chunk_len:int=512
):
    if eq_no_of_chunks_from_ft is not None and eq_chunk_len is not None:
        number_of_chunks = math.ceil((eq_no_of_chunks_from_ft*(eq_chunk_len-overlaps))/(current_chunk_len-overlaps)-4)
    elif number_of_chunks is None:
        print("Returning all documents. As no 'number of chunks' is given.")
        return indexes , embeddings, labels
    
    indexes_ , embeddings_, labels_ = [],[],[]
    if number_of_chunks == 0:
        number_of_chunks = number_of_chunks+1
    for i, (index, embeds, label) in enumerate(zip(indexes, embeddings, labels)):
        if index>=number_of_chunks-1: #-1 to keep right number of retrieved documents when comparing with fine-tuned models for minimum_number_of_chunks.
            indexes_.append(index)
            embeddings_.append(embeds)
            labels_.append(label)
    indexes_ = np.asarray(indexes_, dtype=object)
    embeddings_ = np.asarray(embeddings_, dtype=object)
    labels_ = np.asarray(labels_, dtype=object).astype('int32')
    return indexes_ , embeddings_, labels_

def get_only_large_documents__(number_of_chunks, indexes, embeddings, labels):
    indexes_ , embeddings_, labels_ = [],[],[]
    if number_of_chunks == 0:
        number_of_chunks = number_of_chunks+1
    for i, (index, embeds, label) in enumerate(zip(indexes, embeddings, labels)):
        if index >= number_of_chunks-1: 
            indexes_.append(index)
            embeddings_.append(embeds) 
            labels_.append(label)
    indexes_ = np.asarray(indexes_, dtype=object)
    embeddings_ = np.asarray(embeddings_, dtype=object)
    labels_ = np.asarray(labels_, dtype=object).astype('int32')
    return indexes_ , embeddings_, labels_

def calculate_indexes(embeds):
    indexes = []
    for i in range(len(embeds)):
        index = embeds[i].shape[0]
        indexes.append(index)
        
    return indexes
    
def main_(args):
    #args = args_parser()
    print(args)
    dict_args = vars(args)
    
    #-------------------------------Params to save the trained model--------------------------------------
    if args.combine_layers:
        #extracted_data = f"last_{args.layers_from_end}_layers"
        extracted = f"last_{args.layers_from_end}_layers"
        if args.add_layers:
            extracted = extracted+"_added"
        else:
            extracted = extracted+"_concatenated"
    else:
        #extracted = "last_1_layers/"
        extracted = f"last_layer"

    if args.with_clustering:
        model_name_ = f"{args.dimReduction}_hdbscan"
        category = "Hard_clustered_"
        if args.bilstm_before_encoder:
            category = category+"RNN_to_Encoder_"
        if args.include_rnn:
            category = category+"with_RNNs"
        else:
            category = category+"without_RNNs"    
    else:
        model_name_ = "Without_Clustering"
        category = ""
        if args.bilstm_before_encoder:
            category = "RNN_to_Encoder_"
        if args.include_rnn:
            category = category+"with_RNNs"
        else:
            category = category+"without_RNNs"
    
    
    if args.self_ft_extracted:
        name_of_loaded_model = "_ft_"+os.path.basename(os.path.normpath(args.ft_model_used_for_extraction))
    else:
        name_of_loaded_model = "without_ft"
    name_of_finetuned_model = args.pretrained_model.replace("/","_")
    
    dim_r_cluster_save_path = args.parent_dir + f"{args.dataset_subset}/{name_of_finetuned_model}/ft_on_{args.finetuned_on_input_len}_input_length/{name_of_loaded_model}/{extracted}" + '/'
    
    if args.two_rnn_to_enc:
        num_layers_RNN = 2
    else:
        num_layers_RNN = 1
    if args.minimum_number_of_chunks > 0:
        if args.only_RNN:
            type_of_aggregator = f"RNNs-{num_layers_RNN}layers_min_{args.minimum_number_of_chunks}_chunks"
        else:
            type_of_aggregator = f"StackedEncoder-{args.num_layers}layers_min_{args.minimum_number_of_chunks}_chunks"
        model_save_path = dim_r_cluster_save_path + type_of_aggregator + '/' + category + '/' + model_name_ + '/' + f"run_{args.train_run_number}" + '/'
    else:
        if args.only_RNN:
            type_of_aggregator = f"RNNs-{num_layers_RNN}layers"
        else:
            type_of_aggregator = f"StackedEncoder-{args.num_layers}layers"
        model_save_path = dim_r_cluster_save_path + type_of_aggregator + '/' + category + '/' + model_name_ + '/' + f"run_{args.train_run_number}" + '/'
    if os.path.isdir(model_save_path)==False:
        os.makedirs(model_save_path)
    #if os.path.isdir(dim_r_cluster_save_path)==False:
    #    os.makedirs(dim_r_cluster_save_path)

    # model_cl_pred = tf.keras.models.load_model(model_load_path)
    # ---------------------------------------------Loading data----------------------------------------------------

    # loading the corresponding label for each case in dataset
    
    inputs_train = np.load(args.data_path+"train_.npy", allow_pickle = True)
    labels_train = np.load(args.data_path+"train_labels_.npy", allow_pickle = True).astype('int32')
    if args.minimum_number_of_chunks > 0:
        if os.path.isfile(args.data_path+"train_indexes_.npy"):
            train_indexes = np.load(args.data_path+"train_indexes_.npy", allow_pickle = True)
        else:
            train_indexes = calculate_indexes(inputs_train)
            
    inputs_validate = np.load(args.data_path+"validation_.npy", allow_pickle = True)
    labels_validate = np.load(args.data_path+"validation_labels_.npy", allow_pickle = True).astype('int32')
    if os.path.isfile(args.data_path+"validation_indexes_.npy"):
        validate_indexes = np.load(args.data_path+"validation_indexes_.npy", allow_pickle = True)
    else: 
        validate_indexes = calculate_indexes(inputs_validate)
    #num_labels=len(labels_validate[0])
    #labels_validate = labels_validate
    
    if args.to_test: 
        inputs_test = np.load(args.data_path+"test_.npy", allow_pickle = True)
        labels_test = np.load(args.data_path+"test_labels_.npy", allow_pickle = True).astype('int32')
        if os.path.isfile(args.data_path+"test_indexes_.npy"):
            test_indexes = np.load(args.data_path+"test_indexes_.npy", allow_pickle = True)
        else:
            test_indexes = calculate_indexes(inputs_test)
        if args.to_train==False:
            del inputs_train, labels_train, train_indexes
    #if args.edit_for_ildc:
        #swap train and validate
        #inputs_train, inputs_validate = inputs_validate, inputs_train
        #labels_train, labels_validate = labels_validate, labels_train
        
        #or
        
        #load explanation to test

    if args.minimum_number_of_chunks > 0:
        args.eq_chunk_len = args.finetuned_on_input_len if args.eq_chunk_len is None else args.eq_chunk_len
        if args.to_train:
            train_indexes, inputs_train, labels_train = get_only_large_documents(
                number_of_chunks=args.minimum_number_of_chunks,
                indexes=train_indexes,
                embeddings=inputs_train,
                labels=labels_train,
                eq_no_of_chunks_from_ft=args.eq_no_of_chunks_from_ft,
                eq_chunk_len=args.eq_chunk_len,
                overlaps=args.overlaps,
                current_chunk_len=args.current_chunk_len
                )              
        if args.to_test:
            test_indexes, inputs_test, labels_test = get_only_large_documents(
                number_of_chunks=args.minimum_number_of_chunks,
                indexes=test_indexes,
                embeddings=inputs_test,
                labels=labels_test,
                eq_no_of_chunks_from_ft=args.eq_no_of_chunks_from_ft,
                eq_chunk_len=args.eq_chunk_len,
                overlaps=args.overlaps,
                current_chunk_len=args.current_chunk_len
                )
        validate_indexes, inputs_validate, labels_validate = get_only_large_documents(
            number_of_chunks=args.minimum_number_of_chunks,
            indexes=validate_indexes,
            embeddings=inputs_validate,
            labels=labels_validate,
            eq_no_of_chunks_from_ft=args.eq_no_of_chunks_from_ft,
            eq_chunk_len=args.eq_chunk_len,
            overlaps=args.overlaps,
            current_chunk_len=args.current_chunk_len
            )
    
    if args.combine_layers:
        if args.layers_from_end==1:
            for i in range(inputs_validate.shape[0]):
                inputs_validate[i] = inputs_validate[i][0]
            if args.to_test:
                for i in range(inputs_test.shape[0]):
                    inputs_test[i] = inputs_test[i][0]
            if args.to_train:
                for i in range(inputs_train.shape[0]):
                    inputs_train[i] = inputs_train[i][0]
        elif args.layers_from_end>1 and args.layers_from_end in [1,2,3,4]:
            l_ = args.layers_from_end
            for i in range(inputs_validate.shape[0]):
                inputs_validate[i] = inputs_validate[i][:,:l_,:]
            if args.to_test:
                for i in range(inputs_test.shape[0]):
                    inputs_test[i] = inputs_test[i][:,:l_,:]
            if args.to_train:
                for i in range(inputs_train.shape[0]):
                    inputs_train[i] = inputs_train[i][:,:l_,:]
            
            if args.add_layers:
                for i in range(inputs_validate.shape[0]):
                    inputs_validate[i] = np.sum(inputs_validate[i], axis = 1)
                if args.to_test:
                    for i in range(inputs_test.shape[0]):
                        inputs_test[i] = np.sum(inputs_test[i], axis = 1)
                if args.to_train:
                    for i in range(inputs_train.shape[0]):
                        inputs_train[i] = np.sum(inputs_train[i], axis = 1)

            else:
                for i in range(inputs_validate.shape[0]):
                    inputs_validate[i] = np.reshape(inputs_validate[i], (inputs_validate[i].shape[0], -1))  
                if args.to_test:
                    for i in range(inputs_test.shape[0]):
                        inputs_test[i] = np.reshape(inputs_test[i], (inputs_test[i].shape[0], -1))
                if args.to_train:
                    for i in range(inputs_train.shape[0]):
                        inputs_train[i] = np.reshape(inputs_train[i], (inputs_train[i].shape[0], -1))


    if args.dataset_subset in ["eurlex","ecthr_a","ecthr_b"]:
        type_of_classification = "multi_label"#"multi_class"
        num_labels = len(labels_validate[0])
    elif args.dataset_subset in ["scotus"]:
        type_of_classification = "multi_class"
        num_labels = len(labels_validate[0])
    elif args.dataset_subset in ["ildc"]:
        type_of_classification = "binary"
        num_labels = 1

    # ----------------------------------------Loading data generators-------------------------------------------------------------------------------

    import data_generators
    datagen = data_generators.DataGen(num_labels=num_labels)

    # todo: change the datagen to output two values | status: done
    batches_per_epoch_val =  int(len(labels_validate)/args.batch_size)


    # ----------------------------------------creating model parameters for training-----------------------------------------------------------------

    metrics=['acc',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()]
    if args.to_train:

        batches_per_epoch_train =  int(len(labels_train)/args.batch_size)
        
        if args.with_clustering:
            classification_with_hard_clustering = Classification_with_Clustering(strategy = args.strategy, n_components = 64)
            dim_r_embed_train, dim_r_embed_val, clustered_train, clustered_val = classification_with_hard_clustering.Train_load_DimRed_Clustering(
                inputs_train = inputs_train, 
                inputs_validate = inputs_validate, 
                model_save_path = dim_r_cluster_save_path, 
                train_dimR = args.train_dimR, 
                train_clusterer = args.train_clusterer, 
                save_dimR = args.save_dimR,
                save_clusterer = args.save_clusterer, 
                clustering=args.dimReduction, 
                dim_reduction_metric = args.dim_reduction_metric, 
                pad_len =args.pad_len
                ) #labels_train = labels_train, labels_validate = labels_validate,
            
            if args.to_test:
                dim_r_embed_val_, dim_r_embed_test, clustered_val_, clustered_test = classification_with_hard_clustering.Test_load_DimRed_Clustering(
                    inputs_validate = inputs_validate, 
                    inputs_test = inputs_test,
                    model_save_path = dim_r_cluster_save_path,
                    clustering_strategy = args.clustering_strategy,
                    clustering=args.dimReduction,
                    dim_reduction_metric = args.dim_reduction_metric,
                    pad_len = args.pad_len)
                #del dim_r_embed_val_, clustered_val_
                
            if args.strategy == 2:
                inputs_train = dim_r_embed_train
                inputs_validate = dim_r_embed_val
                del inputs_train, inputs_validate
            dim_r_embed_train_shape = dim_r_embed_train[0].shape
            del dim_r_embed_train, dim_r_embed_val
            num_features=inputs_train[0].shape[1]
            train_datagen = datagen.generator(
                inputs_train,
                labels_train,
                args.batch_size,
                num_features=num_features,
                with_clustering = True,
                clustered_data = clustered_train,
                only_clustered = args.only_clustered
                )
            validation_datagen = datagen.generator(
                inputs_validate,
                labels_validate,
                args.batch_size,
                num_features=num_features,
                with_clustering = True,
                clustered_data = clustered_val,
                only_clustered = args.only_clustered
                )
            """
            if args.to_test:
                
                test_datagen = datagen.generator(
                    inputs_test, 
                    labels_test, 
                    args.batch_size, 
                    num_features=num_features, 
                    with_clustering = True, 
                    clustered_data = clustered_test
                    )
                val_datagen_ = datagen.generator(
                    inputs_validate, 
                    labels_validate, 
                    args.batch_size, 
                    num_features=num_features, 
                    with_clustering = True, 
                    clustered_data = clustered_val_
                    )
            """
                    
        elif not args.with_clustering:
            num_features=inputs_train[0].shape[1] 
            train_datagen = datagen.generator(
                inputs_train, 
                labels_train, 
                args.batch_size, 
                num_features=num_features, 
                with_clustering = False
                )
            validation_datagen = datagen.generator(
                inputs_validate, 
                labels_validate, 
                args.batch_size, 
                num_features=num_features, 
                with_clustering = False
                )
            if args.to_test:
                clustered_test = None
                clustered_val_ = None
                """
                test_datagen = datagen.generator(
                    inputs_test, 
                    labels_test, 
                    args.batch_size, 
                    num_features=num_features, 
                    with_clustering = False, 
                    clustered_data = None
                    )
                val_datagen_ = datagen.generator(
                    inputs_validate, 
                    labels_validate, 
                    args.batch_size, 
                    num_features=num_features, 
                    with_clustering = False, 
                    clustered_data = None
                    )
                """
        if type_of_classification in ['multi_label', 'binary']: #'categorical_crossentropy'
            loss = 'binary_crossentropy'#tf.keras.losses.BinaryCrossentropy()#(from_logits=True)#'binary_crossentropy'
        elif type_of_classification == 'multi_class':
            loss = 'categorical_crossentropy'#tf.keras.losses.CategoricalCrossentropy()#(from_logits=True)#'categorical_crossentropy'
        
        if args.with_clustering:
            model_name = "pUMAP_hdbscan_"
            # Create a single transformer encoder layer.
            TransformerEncoder_ = TransformerEncoder(intermediate_dim=args.dff, num_heads=args.num_heads)

            #encoder = [keras_nlp.layers.TransformerEncoder(intermediate_dim=args.dff, num_heads=args.num_heads) for layer in range(args.num_layers)]
            if args.load_to_retrain==False:
                model__ = Classification_with_clustering_model_(model = TransformerEncoder_, strategy=args.strategy, positional_encoding = args.positional_encoding)
                model_pred = model__.get_encoder_model(
                    CLS_clustered_inputs_shape = inputs_train[0].shape, 
                    clustered_train_shape = clustered_train[0].shape, 
                    dim_r_embed_train_shape = dim_r_embed_train_shape,
                    clustering_strategy = args.clustering_strategy,
                    num_nodes = num_labels, #len(np.asarray(labels_train[0])),
                    intermediate_dim = args.dff,
                    return_training_params = False, 
                    model_save_path = model_save_path, 
                    load = args.load_to_retrain,
                    model_load_path = args.model_load_path,
                    layers_to_freeze = [1], 
                    num_enc_layers=args.num_layers, 
                    include_rnn=args.include_rnn,
                    from_encoder = args.from_encoder,
                    dropout_before_first_fnn = args.dropout_before_first_fnn,
                    dropout_before_inner_fnn = args.dropout_before_inner_fnn,
                    ffn_dropout_value = args.ffn_dropout_value,
                    dropout_after_first_encoder = args.dropout_after_first_encoder,
                    dropout_after_second_encoder = args.dropout_after_second_encoder,
                    encoder_dropout_value = args.encoder_dropout_value,
                    bilstm_before_encoder = args.bilstm_before_encoder,
                    two_rnn_to_enc = args.two_rnn_to_enc,
                    problem_type = type_of_classification,
                    only_RNN = args.only_RNN,
                    num_heads = args.num_heads
                    )
                optimizer = tf.keras.optimizers.Adam(learning_rate=3.5e-6, beta_1=0.9, beta_2=0.98, epsilon=1e-07)
                model_pred.compile(optimizer=optimizer,loss=loss,metrics=metrics)
                model_pred.summary()
                contigious_training = True
            else:
                model_pred = tf.keras.models.load_model(args.model_load_path)
                contigious_training = True
                to_freeze = False

            
            if args.load_to_retrain==False:
                checkpoint_path = model_save_path + f"_{args.clustering_strategy}_strat_{args.strategy}__simple_encoder_{args.num_layers}__dropBeforeFirstFnn_{args.dropout_before_first_fnn}{args.ffn_dropout_value}__dropAfterFirstEnc_{args.dropout_after_first_encoder}{args.encoder_dropout_value}__dropAfterSecondEnc_{args.dropout_after_second_encoder}{args.encoder_dropout_value}__encToRnn_{args.from_encoder}" + "_-{epoch:03d}-{val_loss:.4f}-{val_acc:.4f}"
            else:
                checkpoint_path = model_save_path + f"Encoder_encoder_{args.num_layers}_{args.model_load_path[-15:-25]}_{args.clustering_strategy}_strat_{args.strategy}_" + "-{epoch:03d}-{val_loss:.4f}-{val_acc:.4f}"#freezedEncoder_

        #if to_freeze:
        #    for layer in layers_to_freeze:
        #        model_pred.layers[layer].trainable = False
            print(model_pred.summary())
            lr_scheduler = LearningRateScheduler()
            if args.load_to_retrain==False:# and to_freeze==False:

                #Parameters for Training the model
                warmup_step = 50
                Checkpoint = ModelCheckpoint(filepath=checkpoint_path,save_weights_only=False,monitor='val_acc',mode='max',save_best_only=False)

                if args.save_model_trained_here:
                    callbacks = [lr_scheduler, Checkpoint]
                else:
                    callbacks = [lr_scheduler]
                #h_h = model_pred.fit(train_datagen, steps_per_epoch=batches_per_epoch_train, epochs=args.epochs, validation_data=validation_datagen, validation_steps=batches_per_epoch_val, callbacks = callbacks, verbose = args.verbose)
            
        elif not args.with_clustering:
            if args.load_to_retrain==False:
                TransformerEncoder_ = TransformerEncoder(intermediate_dim=args.dff, num_heads=args.num_heads)
                model__ = Classification_without_clustering_model_(model = TransformerEncoder_, positional_encoding = args.positional_encoding,)
                model_pred = model__.get_encoder_model(
                    inputs_shape = inputs_train[0].shape,
                    num_nodes = num_labels,
                    intermediate_dim = args.dff,
                    return_training_params = False, 
                    model_save_path = model_save_path,
                    load = args.load_to_retrain,
                    model_load_path = args.model_load_path,
                    layers_to_freeze = [1], 
                    num_enc_layers=args.num_layers, 
                    include_rnn=args.include_rnn,
                    from_encoder =args.from_encoder,
                    dropout_before_first_fnn = args.dropout_before_first_fnn,
                    dropout_before_inner_fnn = args.dropout_before_inner_fnn,
                    ffn_dropout_value = args.ffn_dropout_value,
                    dropout_after_first_encoder = args.dropout_after_first_encoder,
                    dropout_after_second_encoder = args.dropout_after_second_encoder,
                    encoder_dropout_value = args.encoder_dropout_value,
                    bilstm_before_encoder = args.bilstm_before_encoder,
                    problem_type = type_of_classification,
                    only_RNN = args.only_RNN,
                    num_heads = args.num_heads
                    )
                
                optimizer = tf.keras.optimizers.Adam(learning_rate=3.5e-6, beta_1=0.9, beta_2=0.98, epsilon=1e-07)
                model_pred.compile(optimizer=optimizer,loss=loss,metrics=metrics)
                model_pred.summary()
                contigious_training = True
            else:
                model_pred = tf.keras.models.load_model(args.model_load_path)
                contigious_training = False
                
            if args.load_to_retrain==False:
                checkpoint_path = model_save_path + f"_simple_encoder_{args.num_layers}__dropBeforeFirstFnn_{args.dropout_before_first_fnn}{args.ffn_dropout_value}__dropAfterFirstEnc_{args.dropout_after_first_encoder}{args.encoder_dropout_value}__dropAfterSecondEnc_{args.dropout_after_second_encoder}{args.encoder_dropout_value}__encToRnn_{args.from_encoder}" + "_-{epoch:03d}-{val_loss:.4f}-{val_acc:.4f}"
            else:
                checkpoint_path = model_save_path + f"freezedEncoder_encoder_{args.num_layers}__{args.model_load_path[-15:-25]}_" + "_-{epoch:03d}-{val_loss:.4f}-{val_acc:.4f}"#freezedEncoder_

            #if to_freeze:
            #    for layer in layers_to_freeze:
            #        model_pred.layers[layer].trainable = False
            print(model_pred.summary())
            lr_scheduler = LearningRateScheduler()
            
            
            if args.load_to_retrain==False:# and to_freeze==False:
                #Parameters for Training the model
                warmup_step = 50
                Checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=False,monitor='val_acc',mode='max',save_best_only=False)
                if args.save_model_trained_here:
                    callbacks = [lr_scheduler, Checkpoint]
                else:
                    callbacks = [lr_scheduler]
                #h_h = model_pred.fit(train_datagen, steps_per_epoch=batches_per_epoch_train, epochs=args.epochs,validation_data=validation_datagen, validation_steps=batches_per_epoch_val, callbacks = callbacks, verbose = args.verbose)
            
        #tf.keras.utils.plot_model(model_pred, to_file=model_save_path+"this_model.png", show_shapes=True)
        dict_args = pd.DataFrame(dict_args).transpose()
        dict_args.to_csv(model_save_path+f"model_args.csv", index= True)
        for epoch in range(args.epochs):
            print(f"epoch {epoch} of {args.epochs} \n")
            """
            checkpoint_path = checkpoint_path + "-{epoch:03d}-{val_loss:.4f}"
            warmup_step = 50
            Checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=False,monitor='val_acc',mode='max',save_best_only=False)
            
            if args.save_model_trained_here:
                callbacks = [lr_scheduler, Checkpoint]
            else:
                callbacks = [lr_scheduler]
            """
            #train for one epoch
            h_h = model_pred.fit(
                train_datagen,
                steps_per_epoch=batches_per_epoch_train,
                epochs=1,
                validation_data=validation_datagen,
                validation_steps=batches_per_epoch_val,
                callbacks = callbacks,
                verbose = args.verbose
                )
            #evaluate on validation
            
            #evaluate on test
            if args.to_test:
                test_datagen = datagen.generator(
                    inputs_test, 
                    labels_test, 
                    args.batch_size, 
                    num_features=inputs_test[0].shape[1], 
                    with_clustering = args.with_clustering, 
                    clustered_data = clustered_test
                    )
                print("\n test--")
                batches_per_epoch_test =  int(len(labels_test)/args.batch_size)
                test_report, test_predictions = predict_plot(
                    model_pred, 
                    labels_test, 
                    data_generator=test_datagen, 
                    batches_per_epoch_test = batches_per_epoch_test, 
                    problem_type=type_of_classification
                    )
                np.save(model_save_path+f"test_predictions_{epoch}.npy", test_predictions)
                test_report.to_csv(model_save_path+f"test_Classification_Report_{epoch}.csv", index= True)
                
                val_datagen_ = datagen.generator(
                    inputs_validate, 
                    labels_validate, 
                    args.batch_size, 
                    num_features=inputs_validate[0].shape[1], 
                    with_clustering = args.with_clustering, 
                    clustered_data = clustered_val_
                    )
                
                print("\n validate--")
                val_report, val_predictions = predict_plot(
                    model_pred, 
                    labels_validate, 
                    data_generator=val_datagen_, 
                    batches_per_epoch_test = batches_per_epoch_val, 
                    problem_type=type_of_classification
                    )
                np.save(model_save_path+f"val_predictions_{epoch}.npy", val_predictions)
                val_report.to_csv(model_save_path+f"val_Classification_Report_{epoch}.csv", index= True)
                
            #continue training
          
        
        print("Training Finished")


    elif args.to_test and args.to_train==False:
        #for multi class metric: https://github.com/tensorflow/addons/issues/1753#issuecomment-1022962370
        print("--------------------------------------------Testing--------------------------------------------------------------")
        #inputs_test = test
        labels_test = np.asarray(labels_test)
        #del test, y_test#, inputs_train, labels_train
        batches_per_epoch_test =  int(len(labels_test)/args.batch_size)
        
        if args.with_clustering:
            classification_with_hard_clustering = Classification_with_Clustering(strategy = args.strategy)
            dim_r_embed_val, dim_r_embed_test, clustered_val, clustered_test = classification_with_hard_clustering.Test_load_DimRed_Clustering(
                inputs_validate = inputs_validate, 
                inputs_test = inputs_test, 
                model_save_path = dim_r_cluster_save_path, 
                clustering_strategy = args.clustering_strategy, 
                clustering=args.dimReduction, 
                dim_reduction_metric = args.dim_reduction_metric, 
                pad_len = args.pad_len
                )
        else:
            clustered_test = None
            clustered_val = None
            
        #if strategy == 1:
        #    inputs_test = inputs_test
        #    inputs_validate = inputs_validate
        if args.strategy == 2:
            inputs_test = dim_r_embed_test
            inputs_validate = dim_r_embed_val
        num_features = inputs_test[0].shape[1]
        #test_datagen = datagen.generator(inputs_test, labels_test, batch_size, num_features=num_features, with_clustering = with_clustering, clustered_data = clustered_test)
        #validation_datagen = datagen.generator(inputs_validate, labels_validate, batch_size, num_features=num_features, with_clustering = with_clustering, clustered_data = clustered_val)

        test_datagen = datagen.generator(
            inputs_test, 
            labels_test, 
            args.batch_size, 
            num_features=num_features, 
            with_clustering = args.with_clustering, 
            clustered_data = clustered_test
            )
        
        validation_datagen = datagen.generator(
            inputs_validate, 
            labels_validate, 
            args.batch_size, 
            num_features=num_features, 
            with_clustering = args.with_clustering, 
            clustered_data = clustered_val
            )
        
        #next(test_datagen)
        #print(test_datagen)
        #model_list = absoluteFilePaths(model_save_path)
        #print(model_list)
        #o = 5
        #print(model_list[o])
        model = tf.keras.models.load_model(args.model_load_path)
        model.summary()
        print(args.model_load_path)
        print("\n test--")
        #predict_plot(model, labels_test, data_generator=test_datagen, batches_per_epoch_test = batches_per_epoch_test, problem_type=type_of_classification)
        batches_per_epoch_test =  int(len(labels_test)/args.batch_size)
        test_report, test_predictions = predict_plot(
            model, 
            labels_test, 
            data_generator=test_datagen, 
            batches_per_epoch_test = batches_per_epoch_test, 
            problem_type=type_of_classification
            )
        np.save(args.model_load_path+'/'+f"test_predictions_.npy", test_predictions)
        test_report.to_csv(args.model_load_path+'/'+f"test_Classification_Report_.csv", index= True)
        print("\n validate--")
        #predict_plot(model, labels_validate, data_generator=validation_datagen, batches_per_epoch_test = batches_per_epoch_val, problem_type=type_of_classification)
        val_report, val_predictions = predict_plot(
            model, 
            labels_validate, 
            data_generator=validation_datagen, 
            batches_per_epoch_test = batches_per_epoch_val, 
            problem_type=type_of_classification
            )
        np.save(args.model_load_path+'/'+f"val_predictions_.npy", val_predictions)
        val_report.to_csv(args.model_load_path+'/'+f"val_Classification_Report_.csv", index= True)

def main():
    args = args_parser()
    if args.minimum_number_of_chunks == None:
        main_(args)
    elif args.eq_no_of_chunks_from_ft is not None: # Running for all minimum_number_of_chunks in [4,6,8,10,12,14,16,18] for finetuned_on_input_len = 512 and, [4,6,8,10] for finetuned_on_input_len = 2048.
        start_chunk = args.start_chunk if args.start_chunk is not None else 0
        for min_num_chunk in [4,6,8,10]:
            if min_num_chunk > start_chunk:
                args.minimum_number_of_chunks = min_num_chunk
                args.eq_no_of_chunks_from_ft = min_num_chunk
                main_(args)
    else: # Running for all minimum_number_of_chunks in [4,6,8,10,12,14,16,18] for finetuned_on_input_len = 512 and, [4,6,8,10] for finetuned_on_input_len = 2048.
        start_chunk = args.start_chunk if args.start_chunk is not None else 0
        if args.finetuned_on_input_len <= 512:
            for min_num_chunk in [4,6,8,10,12,14,16,18,20,22,24,26]:
                if min_num_chunk > start_chunk:
                    args.minimum_number_of_chunks = min_num_chunk
                    main_(args)
        elif args.finetuned_on_input_len <= 2048:
            for min_num_chunk in [4,6,8,10]:
                if min_num_chunk > start_chunk:
                    args.minimum_number_of_chunks = min_num_chunk
                    main_(args)


        
if __name__ == "__main__":
    main()