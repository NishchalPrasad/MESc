from .transformer_encoder_tf import Encoder, positional_encoding
from .keras_nlp_transformer_encoder import TransformerEncoder
from .keras_nlp_positional_encoding import PositionEmbedding
import data_generators
import tensorflow as tf
from tensorflow.keras import layers
#import keras_nlp
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# Dimension reduction and clustering libraries
import umap
from umap.parametric_umap import ParametricUMAP, load_ParametricUMAP
from sklearn.decomposition import PCA
import hdbscan
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import pickle
import tqdm
from tqdm import tqdm


class StackedEncoder(tf.keras.Model):
    def __init__(self, max_positional_encoding, num_layers, dff, d_model, num_heads, batch_size = 16, rate = 0.1, num_nodes=1, activation = 'sigmoid'):
        super().__init__()
        """
        Arguments:
            max_positional_encoding = maximum number of chunks
            num_layers = number of layers in the encoder stack
            dff = dimension for the first dense layer of pointwise feed forward layer
            d_model = dimension of model/input embeddings i.e. 768 here
            num_heads = number of attention heads

        """
        self.Encoder = Encoder(num_layers, dff, d_model, num_heads, batch_size, rate) 
        #Encoder output shape (batch size, sequence length, d_model)
        self.positional_encoding = positional_encoding(max_positional_encoding, d_model)
        #self.input = tf.keras.layers.Input(shape=shape, dtype='float32', name='text_chunks')
        self.maxpooled = tf.keras.layers.GlobalMaxPool1D()
        self.dense_out = tf.keras.layers.Dense(num_nodes, activation = activation)

    def call(self, inputs, mask = None): #trainable = False,
        sequence_len = tf.shape(inputs)[1]
        inputs = inputs #+ self.positional_encoding[:, :sequence_len, :]
        Encoder = self.Encoder(inputs, mask=mask) #trainable,
        maxpooled = self.maxpooled(Encoder)
        dense_out = self.dense_out(maxpooled)

        return dense_out

    
    
class LearningRateScheduler(tf.keras.callbacks.Callback):
    def on_train_batch_start(self, i, batch_logs):
        model.optimizer.lr = dk**(-0.5)*min(i**(-0.5),warmup_step**(-3/2)*i)
        
    
    
class Classification_with_Clustering():
    def __init__(
        self,
        strategy: list = [1,2,3],
        n_components: int = 64,
        min_cluster_size: int = 15,
        batch_size: int = 1,
        data_generator = None,
        n_epochs = None
    ):#, model: tf.keras.Model
        #super().__init__()
        """
        Parameters:
            The 'inputs' to this model are assumed to be of the shape (batches, number of chunks (which can be of variables sizes), 768 (output dimension size from BERT))
            
            
        
        """
        self.strategy = strategy
        self.n_components = n_components
        #self.model = model
        #self.embedder = ParametricUMAP(n_epochs = n_epochs, n_components = n_components, verbose=True) #make verbose = False, if you dont want to log the training for PrarametricUMAP
        #self.embedder = umap.UMAP(n_components=n_components, metric='hellinger')
        self.embedder = ParametricUMAP(n_components = n_components, verbose=True)#, n_training_epochs = 1)
        self.cluster = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=min_cluster_size, prediction_data=True)
        #make soft clustering and shift values by one?
        self.dense_cl = tf.keras.layers.Dense(30,activation = 'softmax')
        self.flatten = tf.keras.layers.Flatten()
        #self.dense_in = tf.keras.layers.Dense(30,activation = 'softmax')
        self.dropout = tf.keras.layers.Dropout(0.08)
        self.dense_out = tf.keras.layers.Dense(1,activation = 'sigmoid')
        self.datagen = data_generators.DataGen()
        self.batch_size = batch_size
        
    def Train_load_DimRed_Clustering(
        self, inputs_train, inputs_validate, 
        #labels_train, labels_validate, 
        model_save_path = None, 
        model_name = None, 
        clustering_strategy = 'hard',
        train_dimR = True, train_clusterer = True,
        save_dimR = False, save_clusterer = False,
        clustering: str = 'pumap',
        dim_reduction_metric: str = 'cosine',
        pad_len: int = 30
        ):
        
        train_document_chunk_sizes = self.log_indices(inputs_train)
        #print(train_document_chunk_sizes)
        inputs_train = np.vstack(inputs_train)
        val_document_chunk_sizes = self.log_indices(inputs_validate)
        #print(val_document_chunk_sizes)
        inputs_validate = np.vstack(inputs_validate)
        slice_train = len(inputs_train)
        slice_validate = len(inputs_validate)
        if train_dimR:
            
            if clustering == 'pumap':
                self.embedder = self.embedder.fit(
                    np.concatenate((inputs_train,inputs_validate),axis=0)
                    )#dim_r_embed_train = self.embedder.fit_transform(inputs_train) 
                #fit embedder on both train and validate 
                del inputs_train, inputs_validate
                dim_r_embed_train = self.embedder.embedding_
                if save_dimR:
                    print("\n saving embedder \n")
                    self.embedder.save(model_save_path+f"embedder_pumap_{self.n_components}dims_metric_{dim_reduction_metric}")
                    print("\n done")
                    print(dim_r_embed_train.shape)
                    
            elif clustering=='umap':
                self.embedder = umap.UMAP(n_components=self.n_components, metric=dim_reduction_metric, verbose=True).fit(np.concatenate((inputs_train,inputs_validate),axis=0))
                del inputs_train, inputs_validate
                dim_r_embed_train = self.embedder.embedding_
                if save_dimR:
                    print("\n saving embedder \n")
                    pickle.dump(self.embedder, open(model_save_path+f"embedder_umap_{self.n_components}dims_metric_{dim_reduction_metric}",'wb'))
                    print("\n done")
                    print(dim_r_embed_train.shape)
                
                
        else:
            if clustering == 'pumap':
                self.embedder = load_ParametricUMAP((model_save_path+f"embedder_pumap_{self.n_components}dims_metric_{dim_reduction_metric}"))
            elif clustering == 'umap':
                self.embedder = pickle.load(open(model_save_path+f"embedder_umap_{self.n_components}dims_metric_{dim_reduction_metric}",'rb'))
            dim_r_embed_train = self.embedder.transform(np.concatenate((inputs_train,inputs_validate),axis=0))
            del inputs_train, inputs_validate

        #embed validation data        
        #dim_r_embed_val = self.embedder.transform(inputs_validate)
        #remove the dim reducing model from memory
        del self.embedder
        
        print(self.cluster)
        if train_clusterer:
            #clusterer = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=15, prediction_data=True).fit(dim_r_embed_train)
            clusterer = self.cluster.fit(dim_r_embed_train)
            
            if save_clusterer:
                print("\n saving cluterer \n")
                pickle.dump(clusterer, open(model_save_path+'clusterer','wb'))
                print("\n done")   
        
        else:
            clusterer = pickle.load(open(model_save_path+'clusterer','rb'))

        
        #y_train = np.asarray(labels_train)
        
        #cluster validation data
        if clustering_strategy == 'soft':
            clustered_train = hdbscan.all_points_membership_vectors(clusterer)[:slice_train]
            #print(clustered_train)
            clustered_val = hdbscan.all_points_membership_vectors(clusterer)[slice_train:]
            #clustered_val = np.asarray(hdbscan.prediction.membership_vector(clusterer, dim_r_embed_val))
        else:
            clustered_train = clusterer.labels_[:slice_train]
            clustered_val = clusterer.labels_[slice_train:]
            #clustered_val = np.asarray(hdbscan.approximate_predict(clusterer, dim_r_embed_val)[0])
        #Chunk the ouputs from clustered_labels and dim_reduced_embed according to document_chunk_sizes
        #to use with model for training.
        #inputs_train = self.rechunk(inputs_train, train_document_chunk_sizes)
        #inputs_validate = self.rechunk(inputs_validate, val_document_chunk_sizes)
        del clusterer
        dim_r_embed_train = self.rechunk(dim_r_embed_train[:slice_train], train_document_chunk_sizes)
        dim_r_embed_val = self.rechunk(dim_r_embed_train[slice_train:],val_document_chunk_sizes)
        #dim_r_embed_val = self.rechunk(dim_r_embed_val, val_document_chunk_sizes)
        
        #pad clustered outputs to maintain a constant input dimension for dense layers
        clustered_train = self.rechunk(clustered_train, train_document_chunk_sizes, pad=True,pad_len=pad_len)
        clustered_val = self.rechunk(clustered_val, val_document_chunk_sizes, pad=True,pad_len=pad_len)
        
        return dim_r_embed_train, dim_r_embed_val, clustered_train, clustered_val
    
    def Test_load_DimRed_Clustering(
        self,
        inputs_validate,
        inputs_test,
        model_save_path = None, 
        clustering_strategy = 'hard',
        clustering: str = 'pumap',
        dim_reduction_metric: str = 'cosine',
        pad_len:int = 30
    ):
        
        #train_document_chunk_sizes = self.log_indices(inputs_train)
        #inputs_train = np.vstack(inputs_train)
        val_document_chunk_sizes = self.log_indices(inputs_validate)
        #print(val_document_chunk_sizes)
        inputs_validate = np.vstack(inputs_validate)
        test_document_chunk_sizes = self.log_indices(inputs_test)
        #print(test_document_chunk_sizes)
        inputs_test = np.vstack(inputs_test)
        
        if clustering == 'pumap':
            self.embedder = load_ParametricUMAP((model_save_path+f"embedder_pumap_{self.n_components}dims_metric_{dim_reduction_metric}"))
        elif clustering == 'umap':
            self.embedder = pickle.load(open(model_save_path+f"embedder_umap_{self.n_components}dims_metric_{dim_reduction_metric}",'rb'))
                  
        clusterer = pickle.load(open(model_save_path+'clusterer','rb'))
        
        #embed and cluster validation data
        dim_r_embed_val = self.embedder.transform(inputs_validate)
        dim_r_embed_test = self.embedder.transform(inputs_test)
        
        if clustering_strategy == 'soft':
            #clustered_train = hdbscan.all_points_membership_vectors(clusterer)
            clustered_val = np.asarray(hdbscan.prediction.membership_vector(clusterer, dim_r_embed_val))
            clustered_test = np.asarray(hdbscan.prediction.membership_vector(clusterer, dim_r_embed_test))
        else:
            #clustered_train = clusterer.labels_
            clustered_val = np.asarray(hdbscan.approximate_predict(clusterer, dim_r_embed_val)[0])
            clustered_test = np.asarray(hdbscan.approximate_predict(clusterer, dim_r_embed_test)[0])
            
        #Chunk the ouputs from clustered_labels and dim_reduced_embed according to document_chunk_sizes
        #to use with model for training.       
        #dim_r_embed_train = self.rechunk(dim_r_embed_train, train_document_chunk_sizes)
        dim_r_embed_val = self.rechunk(dim_r_embed_val, val_document_chunk_sizes,)
        dim_r_embed_test = self.rechunk(dim_r_embed_test, test_document_chunk_sizes)
        
        #pad clustered outputs to maintain a constant input dimension for dense layers
        #clustered_train = self.rechunk(clustered_train, train_document_chunk_sizes, pad=True)
        clustered_val = self.rechunk(clustered_val, val_document_chunk_sizes, pad=True,pad_len=pad_len)
        clustered_test = self.rechunk(clustered_test, test_document_chunk_sizes, pad=True,pad_len=pad_len)
        
        return dim_r_embed_val, dim_r_embed_test, clustered_val, clustered_test 


    
    def log_indices(self, inputs):
        print("\n logging document chunks \n")
        document_indexes = []
        #indexes_wo_dim = []
        for i in tqdm(range(len(inputs))):
            p = inputs[i].shape[0]
            if p == 768:
                document_indexes.append(1)
                #indexes_wo_dim.append(i)
                #train_[i] = np.expand_dims(train[i], axis=0)
            else:
                document_indexes.append(p)
        return document_indexes
    
    
    def rechunk(self, inputs, chunk_sizes, pad=False, pad_len:int = 30):
        print("\n rechunking \n")
        all_docs = []
        all_docs__ = []
        j = 0
        for i in tqdm(range(len(chunk_sizes))):
            all_docs.append(inputs[j:j+chunk_sizes[i]])
            j+=chunk_sizes[i]
        all_docs = np.asarray(all_docs, dtype=object)
        
        print("\n padding \n")
        if pad==True:
            if len(all_docs[0].shape)==2:
                for p in tqdm(range(len(all_docs))):
                    npad = ((0,pad_len-all_docs[p].shape[0]), (0,0))
                    all_docs[p] = np.pad(all_docs[p], pad_width=npad, mode='constant', constant_values=-10)
                return all_docs
            
            else:
                for p in tqdm(range(len(all_docs))):
                    npad = (0,pad_len-all_docs[p].shape[0])
                    #print(f"npad : {npad} \n")
                    #print(f"all_docs[{p}] : {all_docs[p]} \n")
                    if npad[1]<0:
                        #old but working, all_docs[p] = all_docs[p][all_docs[p].shape[0]-pad_len:]
                        all_docs__.append(all_docs[p][all_docs[p].shape[0]-pad_len:])
                    else:
                        #s = np.asarray(all_docs[p])
                        s = np.pad(all_docs[p], pad_width=npad, mode='constant', constant_values=-100000)
                        all_docs__.append(s)
                        #old causing broadcast error, all_docs[p] = np.pad(all_docs[p], pad_width=npad, mode='constant', constant_values=-100000)
                    #print(all_docs[p].shape[0])
                return np.asarray(all_docs__, dtype=object)#all_docs
            
        return all_docs
    
    # changing shape from (k, 1, 768) to (k, 768)
    def change_shape(self, array):
        print("\n changing input shpe of [CLS] embeddings \n")
        for i,t in enumerate(array):
            array[i] = tf.squeeze(t)
        return array
    
    def test(self):
        pass
    
    
class Classification_with_clustering_model_():
    def __init__(
        self,
        model,
        strategy: int = 1,
        batch_size: int = 1,
        positional_encoding: bool = False
    ):
        super().__init__()
                #super().__init__()
        """
        Parameters:
            The 'inputs' to this model are assumed to be of the shape (batches, number of chunks (which can be of variables sizes), 768 (output dimension size from BERT))
            
            
        
        """
        self.strategy = strategy
        self.model = model
        self.batch_size = batch_size
        self.positional_encoding = positional_encoding
        
    def get_encoder_model(
        self,
        CLS_clustered_inputs_shape,
        clustered_train_shape,
        dim_r_embed_train_shape,
        clustering_strategy: str,
        num_nodes: int = 1,
        intermediate_dim: int = 2048,
        return_training_params: bool = False,
        model_save_path: str = None,
        model_name: str = None,
        load: bool = False,
        model_load_path = None,
        layers_to_freeze: list = [1],
        num_enc_layers: int = 1,
        include_rnn: bool = False,
        rnn_name: str = 'bilstm',
        dropout_before_first_fnn: bool = False,
        dropout_before_inner_fnn: bool = False,
        ffn_dropout_value:float = 0.08,
        from_encoder: bool = False,
        dropout_after_first_encoder: bool = False,
        dropout_after_second_encoder: bool = False,
        encoder_dropout_value: float = 0.1,
        bilstm_before_encoder: bool = False,
        two_rnn_to_enc: bool = False,
        problem_type: str = 'multi_label',
        only_RNN: bool = False,
        num_heads: int = 8
    ):
        """
        Parameters:
            The 'inputs' to this model are assumed to be of the shape (batches, number of chunks (which can be of variables sizes),  (output dimension size from BERT))

        """
        #model_ = [self.model for layer in range(num_enc_layers)]
        if problem_type == 'multi_class':
            activation_ = 'softmax'
        else:
            activation_ = 'sigmoid'
        strat = self.strategy
        #for strat in self.strategy:
        print(f"\n Compiling model for strategy {strat}")
        #tf.keras.backend.clear_session()
        if strat == 1 or strat == 2:
            if strat == 1:
                print("""
                    Chunked CLS
                    embeddings  ------------------------> Encoder Model -> output
                        |                                   |
                Dimensionality Reduction --> Clustering ____|
                   (PrametricUMAP)
                """)
                inputs = tf.keras.layers.Input(shape=(None,CLS_clustered_inputs_shape[1],), dtype='float32', name='cls_chunks_')
                clustered = tf.keras.layers.Input(shape=clustered_train_shape, dtype='float32', name='clustered_chunks_')
                seq_len = CLS_clustered_inputs_shape[1]

            elif strat ==2:
                print("""
                    Chunked CLS
                    embeddings           ---------------> Encoder Model -> output
                        |                |                  |
                Dimensionality Reduction --> Clustering ____|
                   (PrametricUMAP)
                """)
                inputs = tf.keras.layers.Input(shape=(None,dim_r_embed_train_shape[1],), dtype='float32', name='dim_reduced_chunks')
                clustered = tf.keras.layers.Input(shape=clustered_train_shape, dtype='float32', name='clustered_chunks_')
                seq_len = dim_r_embed_train_shape[1]
            
            print_st = f"""
                                              Encoder ----> Max pooled ---> dense ---> 
                               {num_enc_layers} layers                                |
                       Encoder Model =                                                |-->dense-->output
                                                                                      |
                                            Clustering ----> dense ------------------>
                    """
            #todo: change the datagen to output two values | status: done
        
            if load:
                model_cl_pred = tf.keras.models.load_model(model_load_path)
            else:
                if self.positional_encoding and only_RNN==False:
                    seq_len = seq_len
                    position_embeddings = PositionEmbedding(
                                            sequence_length=seq_len
                                            )(inputs)
                    inputs = inputs + position_embeddings
                    
                if bilstm_before_encoder == False and only_RNN == False:
                    if num_enc_layers == 1:
                        enc_o = self.model(inputs)
                    elif num_enc_layers == 2:# and strat == 1:
                        encoder_1 = TransformerEncoder(intermediate_dim=intermediate_dim, num_heads=num_heads)
                        encoder_2 = TransformerEncoder(intermediate_dim=intermediate_dim, num_heads=num_heads)
                        enc_1_o = encoder_1(inputs)
                        if dropout_after_first_encoder:
                            enc_1_o = tf.keras.layers.Dropout(encoder_dropout_value)(enc_1_o)#(0.1)
                        enc_o = encoder_2(enc_1_o)
                    elif num_enc_layers == 3:# and strat == 1:
                        encoder_1 = TransformerEncoder(intermediate_dim=intermediate_dim, num_heads=num_heads)
                        encoder_2 = TransformerEncoder(intermediate_dim=intermediate_dim, num_heads=num_heads)
                        encoder_3 = TransformerEncoder(intermediate_dim=intermediate_dim, num_heads=num_heads)
                        enc_1_o = encoder_1(inputs)
                        if dropout_after_first_encoder:
                            enc_1_o = tf.keras.layers.Dropout(encoder_dropout_value)(enc_1_o)#(0.1)
                        enc_2_o = encoder_2(enc_1_o)
                        if dropout_after_second_encoder:
                            enc_2_o = tf.keras.layers.Dropout(encoder_dropout_value)(enc_2_o)
                        enc_o = encoder_3(enc_2_o)   
                    if include_rnn:

                        print_st = f"""#{num_enc_layers} layers                            
                                                  Encoder ----> Max pooled ---> dense ---> 
                                                    |                                     |
                           Encoder Model =        RNN ------> dense --------------------->|-->dense-->output
                                                                                          |
                                                Clustering ----> dense ------------------>
                           """
                        if from_encoder:
                            # Masking layer to mask the padded values
                            enc_o_r = layers.Masking(mask_value=-99.)(enc_o)
                        else:
                            # Masking layer to mask the padded values
                            enc_o_r = layers.Masking(mask_value=-99.)(inputs)
                        # After masking we encoded the vector using 2 bidirectional RNN's
                        enc_o_r = layers.Bidirectional(layers.LSTM(100,return_sequences=True))(enc_o_r)
                        enc_o_r = layers.Bidirectional(layers.LSTM(100))(enc_o_r)
                        # Added a dense layer after encoding
                        enc_o_r = layers.Dense(30, activation='relu')(enc_o_r)

                    maxpooled = tf.keras.layers.GlobalMaxPool1D()(enc_o)
                    if dropout_before_inner_fnn or num_enc_layers >1 or include_rnn:
                        maxpooled = tf.keras.layers.Dropout(ffn_dropout_value)(maxpooled)
                    dense_out = tf.keras.layers.Dense(128, activation = 'relu')(maxpooled)

                    dense_cl = tf.keras.layers.Dense(64,activation = 'softmax')(clustered) #30 was old value, 
                    #dense_cl = tf.keras.layers.Flatten()(dense_cl)

                    if dropout_before_first_fnn or num_enc_layers >1 or include_rnn:
                        dense_out = tf.keras.layers.Dropout(ffn_dropout_value)(dense_out) #(0.08)

                    if include_rnn:
                        conc = tf.keras.layers.concatenate([dense_out,enc_o_r,dense_cl])
                    else:
                        conc = tf.keras.layers.concatenate([dense_out,dense_cl])

                    #if num_enc_layers >1 or include_rnn:
                    #    conc = tf.keras.layers.Dropout(ffn_dropout_value)(conc)
                    pred = tf.keras.layers.Dense(num_nodes,activation = activation_)(conc)

                elif bilstm_before_encoder and only_RNN == False:
                    # Masking layer to mask the padded values
                    enc_o_r = layers.Masking(mask_value=-99.)(inputs)
                    enc_o_r = tf.keras.layers.Bidirectional(layers.LSTM(100,return_sequences=True))(enc_o_r)
                    if two_rnn_to_enc==True:
                        enc_o_r = layers.Bidirectional(layers.LSTM(100),return_sequences=True)(enc_o_r)
                        
                    if num_enc_layers == 1:
                        encoder_1 = TransformerEncoder(intermediate_dim=intermediate_dim, num_heads=num_heads)
                        enc_o = encoder_1(enc_o_r)
                    elif num_enc_layers == 2:# and strat == 1:
                        encoder_1 = TransformerEncoder(intermediate_dim=intermediate_dim, num_heads=num_heads)
                        encoder_2 = TransformerEncoder(intermediate_dim=intermediate_dim, num_heads=num_heads)
                        enc_1_o = encoder_1(enc_o_r)
                        if dropout_after_first_encoder:
                            enc_1_o = tf.keras.layers.Dropout(encoder_dropout_value)(enc_1_o)#(0.1)
                        enc_o = encoder_2(enc_1_o)
                    elif num_enc_layers == 3:# and strat == 1:
                        encoder_1 = TransformerEncoder(intermediate_dim=intermediate_dim, num_heads=num_heads)
                        encoder_2 = TransformerEncoder(intermediate_dim=intermediate_dim, num_heads=num_heads)
                        encoder_3 = TransformerEncoder(intermediate_dim=intermediate_dim, num_heads=num_heads)
                        enc_1_o = encoder_1(enc_o_r)
                        if dropout_after_first_encoder:
                            enc_1_o = tf.keras.layers.Dropout(encoder_dropout_value)(enc_1_o)#(0.1)
                        enc_2_o = encoder_2(enc_1_o)
                        if dropout_after_second_encoder:
                            enc_2_o = tf.keras.layers.Dropout(encoder_dropout_value)(enc_2_o)
                        enc_o = encoder_3(enc_2_o)   
                    
                    if include_rnn:

                        print_st = f"""#{num_enc_layers} layers                            
                                                  Encoder ----> Max pooled ---> dense ---> 
                                                    |                                     |
                           Encoder Model =        RNN ------> dense --------------------->|-->dense-->output
                                                                                          |
                                                Clustering ----> dense ------------------>
                           """
                        if from_encoder:
                            # Masking layer to mask the padded values
                            enc_o_rn = layers.Masking(mask_value=-99.)(enc_o)
                        else:
                            # Masking layer to mask the padded values
                            enc_o_rn = layers.Masking(mask_value=-99.)(enc_o_r)
                        # After masking we encoded the vector using 2 bidirectional RNN's
                        #enc_o_rn = layers.Bidirectional(layers.LSTM(100,return_sequences=True))(enc_o_rn)
                        enc_o_rn = layers.Bidirectional(layers.LSTM(100))(enc_o_rn)
                        # Added a dense layer after encoding
                        enc_o_rn = layers.Dense(30, activation='relu')(enc_o_rn)

                    maxpooled = tf.keras.layers.GlobalMaxPool1D()(enc_o)
                    if dropout_before_inner_fnn or num_enc_layers >1 or include_rnn:
                        maxpooled = tf.keras.layers.Dropout(ffn_dropout_value)(maxpooled)
                    dense_out = tf.keras.layers.Dense(256, activation = 'relu')(maxpooled)

                    dense_cl = tf.keras.layers.Dense(64,activation = 'softmax')(clustered)
                    dense_cl = tf.keras.layers.Flatten()(dense_cl)



                    if dropout_before_first_fnn or num_enc_layers >1 or include_rnn:
                        dense_out = tf.keras.layers.Dropout(ffn_dropout_value)(dense_out) #(0.08)


                    if include_rnn:
                        conc = tf.keras.layers.concatenate([dense_out,enc_o_rn,dense_cl])
                    else:
                        conc = tf.keras.layers.concatenate([dense_out,dense_cl])

                    if num_enc_layers >1 or include_rnn:
                        conc = tf.keras.layers.Dropout(ffn_dropout_value)(conc)
                    pred = tf.keras.layers.Dense(num_nodes,activation = activation_)(conc)    
                    #pred = tf.keras.layers.Dense(num_nodes,activation = 'sigmoid')(conc)
                    
                elif only_RNN:
                    enc_o_r = layers.Masking(mask_value=-99.)(inputs)
                    if two_rnn_to_enc:
                        return_sequences = True
                    else:
                        return_sequences = False
                    enc_o_r = tf.keras.layers.Bidirectional(layers.LSTM(100,return_sequences=return_sequences))(enc_o_r)
                    if two_rnn_to_enc==True:
                        enc_o_r = layers.Bidirectional(layers.LSTM(100))(enc_o_r)
                    # Added a dense layer after encoding
                    enc_o_r = tf.keras.layers.Dropout(0.25)(enc_o_r)
                    out_dense = layers.Dense(64, activation='relu')(enc_o_r)
                    
                    #clustered inputs
                    dense_cl = tf.keras.layers.Dense(64,activation = 'softmax')(clustered)
                    dense_cl = tf.keras.layers.Flatten()(dense_cl)
                    
                    
                    #if dropout_before_first_fnn or num_enc_layers >1 or include_rnn:
                    #    dense_out = tf.keras.layers.Dropout(ffn_dropout_value)(dense_out) #(0.08)
                    
                    conc = tf.keras.layers.concatenate([out_dense,dense_cl])
                    conc = tf.keras.layers.Dropout(0.25)(conc)
                    pred = tf.keras.layers.Dense(num_nodes,activation = activation_)(conc)
                    
                    
                #create model to return    
                model_cl_pred = tf.keras.Model([inputs, clustered], pred)
                print(print_st)
            
        elif strat == 3:
            """
                Chunked CLS
                embeddings                            Model -> output
                    |                                   |
            Dimensionality Reduction --> Clustering ____|
               (PrametricUMAP)
            """
            print(f"this strategy={self.strategy} is not implimented")
            clustered = tf.keras.layers.Input(shape=clustered_train_shape, dtype='float32', name='clustered_chunks_')
            if load:
                model_cl_pred = tf.keras.models.load_model(model_load_path)
            else:        
                dense_cl = tf.keras.layers.Dense(30,activation = 'softmax')(clustered)
                dense_cl = tf.keras.layers.Flatten()(dense_cl)
                conc = tf.keras.layers.Dropout(0.08)(dense_cl)
                pred = tf.keras.layers.Dense(1,activation = activation_)(conc)
                
                model_cl_pred = tf.keras.Model(clustered, pred)
        else:
            print(f"this strategy={self.strategy} is not implimented")
            
        if load:
            for layer in layers_to_freeze:
                model_cl_pred.layers[layer].trainable = False

        
        if return_training_params:
            #Parameters for Training the model
            warmup_step = 50
            lr_scheduler = LearningRateScheduler()
            # Setting the callback
            #call_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.95, patience=2, verbose=2,
            #                                mode='auto', min_delta=0.01, cooldown=0, min_lr=0)
            checkpoint_path = model_save_path + model_name + f"_{clustering_strategy}_strat_{strat}_" + "trained-{epoch:03d}-{val_loss:.4f}-{val_acc:.4f}"
            Checkpoint = ModelCheckpoint(filepath=checkpoint_path,save_weights_only=False,monitor='val_acc',mode='max',save_best_only=False)
            #EarlyStopping = EarlyStopping(monitor='val_loss', patience=3)

            callbacks =[lr_scheduler, Checkpoint]

            return model_cl_pred, callbacks
        
        
        return model_cl_pred

class Classification_without_clustering_model_():
    def __init__(
        self,
        model,
        batch_size: int = 1,
        positional_encoding: bool = False
    ):
        super().__init__()
        """
        Parameters:
            The 'inputs' to this model are assumed to be of the shape (batches, number of chunks (which can be of variables sizes), 768 (output dimension size from BERT))
        
        """
        self.model = model
        self.batch_size = batch_size
        self.positional_encoding = positional_encoding
        
    def get_encoder_model(
        self,
        inputs_shape,
        num_nodes: int = 1,
        intermediate_dim: int = 2048,
        return_training_params: bool = False,
        model_save_path: str = None,
        model_name: str = None,
        load: bool = False,
        model_load_path = None,
        layers_to_freeze: list = [1],
        num_enc_layers: int = 1,
        include_rnn: bool = False,
        rnn_name: str = 'bilstm',
        dropout_before_first_fnn: bool = False,
        dropout_before_inner_fnn: bool = False,
        ffn_dropout_value:float = 0.08,
        from_encoder: bool = True,
        dropout_after_first_encoder: bool = False,
        dropout_after_second_encoder: bool = False,
        encoder_dropout_value: float = 0.1,
        bilstm_before_encoder: bool = False,
        two_rnn_to_enc: bool = False,
        problem_type: str = 'multi_label',
        only_RNN: bool = False,
        num_heads: int = 8
    ):
        """
        Parameters:
            The 'inputs' to this model are assumed to be of the shape (batches, number of chunks (which can be of variables sizes),  (output dimension size from BERT))

        """
        if problem_type == 'multi_class':
            activation_ = 'softmax'
        else:
            activation_ = 'sigmoid'
        print(f"\n Compiling model")
        #tf.keras.backend.clear_session()
        print("""
            Chunked CLS
            embeddings  ------------------------> Encoder Model -> output
        """)
        inputs = tf.keras.layers.Input(shape=(None,inputs_shape[1],), dtype='float32', name='cls_chunks_')
        print_st = f"""
                                          Encoder ----> Max pooled ---> dense ---> 
                                    {num_enc_layers}layers                        |
                   Encoder Model =                                                |-->dense-->output
                                                                                  
                """
        #todo: change the datagen to output two values | status: done
        if load:
            model_cl_pred = tf.keras.models.load_model(model_load_path)
        else:
            if only_RNN==False:
                if self.positional_encoding:
                    seq_len = inputs_shape[1]
                    position_embeddings = PositionEmbedding(
                                                    sequence_length=seq_len
                                                    )(inputs)
                    inputs = inputs + position_embeddings
                #inputs = tf.keras.layers.Normalization(axis=-1, mean=0, variance=1)(inputs)
                if num_enc_layers == 1:
                    enc_o = self.model(inputs)
                elif num_enc_layers == 2:# and strat == 1:
                    encoder_1 = TransformerEncoder(intermediate_dim=intermediate_dim, num_heads=num_heads)
                    encoder_2 = TransformerEncoder(intermediate_dim=intermediate_dim, num_heads=num_heads)
                    enc_1_o = encoder_1(inputs)
                    if dropout_after_first_encoder:
                        enc_1_o = tf.keras.layers.Dropout(encoder_dropout_value)(enc_1_o)
                    enc_o = encoder_2(enc_1_o)
                elif num_enc_layers == 3:# and strat == 1:
                    encoder_1 = TransformerEncoder(intermediate_dim=intermediate_dim, num_heads=num_heads)
                    encoder_2 = TransformerEncoder(intermediate_dim=intermediate_dim, num_heads=num_heads)
                    encoder_3 = TransformerEncoder(intermediate_dim=intermediate_dim, num_heads=num_heads)
                    enc_1_o = encoder_1(inputs)
                    if dropout_after_first_encoder:
                        enc_1_o = tf.keras.layers.Dropout(encoder_dropout_value)(enc_1_o)#(0.1)
                    enc_2_o = encoder_2(enc_1_o)
                    if dropout_after_second_encoder:
                        enc_2_o = tf.keras.layers.Dropout(encoder_dropout_value)(enc_2_o)
                    enc_o = encoder_3(enc_2_o)  
                if include_rnn:

                    print_st = f"""{num_enc_layers} layers                            
                                              Encoder ----> Max pooled ---> dense ---> 
                                                |                                     |
                       Encoder Model =        RNN ------> dense --------------------->|-->dense-->output
                                                                                      |
                                            Clustering ----> dense ------------------>
                    """
                    if from_encoder:
                        # Masking layer to mask the padded values
                        enc_o_r = layers.Masking(mask_value=-99.)(enc_o)
                    else:
                        # Masking layer to mask the padded values
                        enc_o_r = layers.Masking(mask_value=-99.)(inputs)
                    # After masking we encoded the vector using 2 bidirectional RNN's
                    enc_o_r = layers.Bidirectional(layers.LSTM(100,return_sequences=True))(enc_o_r)
                    enc_o_r = layers.Bidirectional(layers.LSTM(100))(enc_o_r)
                    # Added a dense layer after encoding
                    enc_o_r = layers.Dense(30, activation='relu')(enc_o_r)

                maxpooled = tf.keras.layers.GlobalMaxPool1D()(enc_o)
                if dropout_before_inner_fnn or num_enc_layers >1 or include_rnn:
                    maxpooled = tf.keras.layers.Dropout(ffn_dropout_value)(maxpooled)
                dense_out = tf.keras.layers.Dense(128, activation = 'relu')(maxpooled)

                if dropout_before_first_fnn and (num_enc_layers >1 or include_rnn):
                    dense_out = tf.keras.layers.Dropout(ffn_dropout_value)(dense_out)

                if include_rnn:
                    dense_out = tf.keras.layers.concatenate([dense_out,enc_o_r])

                if num_enc_layers >1 or include_rnn:
                    dense_out = tf.keras.layers.Dropout(ffn_dropout_value)(dense_out)
                pred = tf.keras.layers.Dense(num_nodes,activation = activation_)(dense_out)
                
            else:
                print_st = f"""
                       Encoder Model = RNN ------> dense ------->dense-->output
                    """
                enc_o_r = layers.Masking(mask_value=-99.)(inputs)
                if two_rnn_to_enc:
                    return_sequences = True
                else:
                    return_sequences = False
                enc_o_r = tf.keras.layers.Bidirectional(layers.LSTM(100,return_sequences=return_sequences))(enc_o_r)
                if two_rnn_to_enc==True:
                    enc_o_r = layers.Bidirectional(layers.LSTM(100))(enc_o_r)
                    # Added a dense layer after encoding
                enc_o_r = tf.keras.layers.Dropout(0.25)(enc_o_r)
                out_dense = layers.Dense(64, activation='relu')(enc_o_r)
                pred = tf.keras.layers.Dense(num_nodes,activation = activation_)(out_dense)
                    
            model_pred = tf.keras.Model(inputs, pred)

            print(print_st)

        #if load == False:
        #    optimizer = tf.keras.optimizers.Adam(learning_rate=3.5e-6, beta_1=0.9, beta_2=0.98, epsilon=1e-07)
        #    model_cl_pred.compile(optimizer=optimizer,
        #                          loss='binary_crossentropy',
        #                          metrics=['acc'])
        #model_cl_pred.summary()

        if load:
            for layer in layers_to_freeze:
                model_cl_pred.layers[layer].trainable = False

        """
        if return_training_params:
            #Parameters for Training the model
            warmup_step = 50
            lr_scheduler = LearningRateScheduler()
            # Setting the callback
            #call_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.95, patience=2, verbose=2,
            #                                mode='auto', min_delta=0.01, cooldown=0, min_lr=0)
            checkpoint_path = model_save_path + model_name + f"_{clustering_strategy}_strat_{strat}_" + "trained-{epoch:03d}-{val_loss:.4f}-{val_acc:.4f}"
            Checkpoint = ModelCheckpoint(filepath=checkpoint_path,save_weights_only=False,monitor='val_acc',mode='max',save_best_only=False)
            #EarlyStopping = EarlyStopping(monitor='val_loss', patience=3)

            callbacks =[lr_scheduler, Checkpoint]

            return model_pred, callbacks
        """
        
        return model_pred
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
"""    
    if include_rnn:

        print_st = f'{num_enc_layers} layers                            
                                  Encoder ----> Max pooled ---> dense ---> 
                                    |                                     |
           Encoder Model =        RNN ------> dense --------------------->|-->dense-->output
                                                                          |
                                Clustering ----> dense ------------------>
        '
        if from_encoder:
            # Masking layer to mask the padded values
            enc_o_r = layers.Masking(mask_value=-99.)(enc_o)
        else:
            # Masking layer to mask the padded values
            enc_o_r = layers.Masking(mask_value=-99.)(inputs)
        # After masking we encoded the vector using 2 bidirectional RNN's
        enc_o_r = layers.Bidirectional(layers.LSTM(100,return_sequences=True))(enc_o_r)
        enc_o_r = layers.Bidirectional(layers.LSTM(100))(enc_o_r)
        # Added a dense layer after encoding
        enc_o_r = layers.Dense(30, activation='relu')(enc_o_r)

    maxpooled = tf.keras.layers.GlobalMaxPool1D()(enc_o)
    dense_out = tf.keras.layers.Dense(128, activation = 'relu')(maxpooled)

    dense_cl = tf.keras.layers.Dense(30,activation = 'softmax')(clustered)
    dense_cl = tf.keras.layers.Flatten()(dense_cl)
"""



"""def train(self, inputs_train, inputs_validate, labels_train, labels_validate, model_save_path = None, model_name = None, clustering_strategy = 'hard'):
        train_document_chunk_sizes = self.log_indices(inputs_train)
        inputs_train = np.vstack(inputs_train)
        val_document_chunk_sizes = self.log_indices(inputs_validate)
        inputs_validate = np.vstack(inputs_validate)
        
        dim_r_embed_train = self.embedder.fit_transform(inputs_train)
        clusterer = self.cluster.fit(dim_r_embed_train)
            
        #save embedder and clustering algorithm
        print("\n saving embedder \n")
        self.embedder.save(model_save_path+'embedder')
        print("\n done")
        print("\n saving cluterer \n")
        pickle.dump(clusterer, open(model_save_path+'clusterer','wb'))
        print("\n done")
        #embed and cluster validation data
        dim_r_embed_val = self.embedder.transform(inputs_validate)
        
        if clustering_strategy == 'soft':
            clustered_train = hdbscan.all_points_membership_vectors(clusterer)
            clustered_val = hdbscan.prediction.membership_vector(clusterer, dim_r_embed_val)
        else:
            clustered_train = clusterer.labels_
            clustered_val = np.asarray(hdbscan.approximate_predict(clusterer, dim_r_embed_val)[0])
        #Chunk the ouputs from clustered_labels and dim_reduced_embed according to document_chunk_sizes
        #to use with model for training.
        #inputs_train = self.rechunk(inputs_train, train_document_chunk_sizes)
        #inputs_validate = self.rechunk(inputs_validate, val_document_chunk_sizes)
        
        dim_r_embed_train = self.rechunk(dim_r_embed_train, train_document_chunk_sizes)
        dim_r_embed_val = self.rechunk(dim_r_embed_val, val_document_chunk_sizes)
        
        #pad clustered outputs to maintain a constant input dimension for dense layers
        clustered_train = self.rechunk(clustered_train, train_document_chunk_sizes, pad=True)
        clustered_val = self.rechunk(clustered_val, val_document_chunk_sizes, pad=True)
        
        
        for strat in self.strategy:
            print(f"\n Training for strategy {strat}")
            tf.keras.backend.clear_session()
            if strat == 1 or strat == 2:
                if strat == 1:
                    print(""""""
                        Chunked CLS
                        embeddings  ------------------------> Model -> output
                            |                                   |
                    Dimensionality Reduction --> Clustering ____|
                       (PrametricUMAP)
                    """""")
                    inputs = tf.keras.layers.Input(shape=(None,768,), dtype='float32', name='text_chunks')
                    clustered = tf.keras.layers.Input(shape=clustered_train[0].shape, dtype='float32', name='text_chunks')
                    train_datagen = self.datagen.generator(self.change_shape(inputs_train), labels_train, self.batch_size, with_clustering = True, clustered_data = clustered_train)
                    validation_datagen = self.datagen.generator(self.change_shape(inputs_validate), labels_validate, self.batch_size, with_clustering = True, clustered_data = clustered_val)
                    
                elif strat ==2:
                    print(""""""
                        Chunked CLS
                        embeddings           ---------------> Model -> output
                            |                |                  |
                    Dimensionality Reduction --> Clustering ____|
                       (PrametricUMAP)
                    """""")
                    inputs = tf.keras.layers.Input(shape=(None,n_components,), dtype='float32', name='text_chunks')
                    clustered = tf.keras.layers.Input(shape=clustered_train[0].shape, dtype='float32', name='text_chunks')
                    train_datagen = self.datagen.generator(dim_r_embed_train, labels_train, self.batch_size, with_clustering = True, clustered_data = clustered_train)
                    validation_datagen = self.datagen.generator(dim_r_embed_val, labels_validate, self.batch_size, with_clustering = True, clustered_data = clustered_val)
                    
                    
                #todo: change the datagen to output two values | status: done
                model_o = model(inputs)
                dense_cl = self.dense_cl(clustered)
                dense_cl = self.flatten(dense_cl)
                pred = self.dense_out([model_o,dense_cl])
                model_cl_pred = tf.keras.Model([inputs, clustered], pred)

                
            elif strat == 3:
                """"""
                    Chunked CLS
                    embeddings                            Model -> output
                        |                                   |
                Dimensionality Reduction --> Clustering ____|
                   (PrametricUMAP)
                """"""
                print(f"this strategy={self.strategy} is not implimented")
                break
            else:
                print(f"this strategy={self.strategy} is not implimented")
                break

            
            optimizer = tf.keras.optimizers.Adam(learning_rate=3.5e-6, beta_1=0.9, beta_2=0.98, epsilon=1e-07)
            model_cl_pred.compile(optimizer=optimizer,
                                  loss='binary_crossentropy',
                                  metrics=['acc'])
            model_cl_pred.summary()
            
            
            #Parameters for Training the model
            warmup_step = 50
            lr_scheduler = LearningRateScheduler()
            # Setting the callback
            #call_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.95, patience=2, verbose=2,
            #                                mode='auto', min_delta=0.01, cooldown=0, min_lr=0)
            checkpoint_path = model_save_path + model_name + f"_{clustering_strategy}_strat_{strat}" + "trained-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}"
            Checkpoint = ModelCheckpoint(filepath=checkpoint_path,save_weights_only=False,monitor='val_acc',mode='max',save_best_only=False)
            EarlyStopping = EarlyStopping(monitor='val_loss', patience=3)
            
            batches_per_epoch_train =  int(len(labels_train)/self.batch_size)
            batches_per_epoch_val =  int(len(labels_validate)/self.batch_size)
            
            model.fit_generator(train_generator, steps_per_epoch=batches_per_epoch_train, epochs=5,
                                validation_data=val_generator, validation_steps=batches_per_epoch_val, callbacks =[lr_scheduler, Checkpoint], verbose = 1)
        
    """        