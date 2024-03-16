import numpy as np

class DataGen():
    def __init__(self, num_labels: int = 1):
        super().__init__()
        self.num_labels = num_labels
        
    def generator(self, data, labels, batch_size, num_features = 768, test=False, with_clustering = False, clustered_data = None, only_clustered: bool = False):
        num_sequences = len(data) 
        batches_per_epoch = int(num_sequences/batch_size)
        if test:
            batches_per_epoch+=1
        x_list = data
        y_list = labels
        if with_clustering:
            x_clustered = clustered_data
        # Generate batches
        while True:
            for b in range(batches_per_epoch):
                if(test & b == batches_per_epoch-1): # An extra if else statement just to manage the last batch as it's size might not be equal to batch size 
                    #print("last batch")
                    longest_index = num_sequences - 1
                    timesteps = len(max(x_list[:longest_index + 1][-batch_size:], key=len))
                    _batch_size = longest_index - b*batch_size
                else:
                    #print("intermediate batch")
                    longest_index = (b + 1) * batch_size - 1
                    timesteps = len(max(x_list[:(b + 1) * batch_size][-batch_size:], key=len))
                    _batch_size = batch_size
                
                x_ = np.full((_batch_size, timesteps, num_features), -99.)
                y_ = np.zeros((_batch_size, self.num_labels))
                
                #delete this after testing, gives error during training when shape is 2 dimensional
                #x_cl = np.full((_batch_size, x_clustered[0].shape[0]), -99.)
                
                #for clustered data
                #considering shaped of clustered data is (number of documents,1) and for each document, shape = (number of chunks+padded = 30, number of features)
                if with_clustering and len(x_clustered[0].shape) == 2:
                    x_cl = np.full((_batch_size, x_clustered[0].shape[0], x_clustered[0].shape[1]), -99.)
                elif with_clustering and len(x_clustered[0].shape) == 1:
                    x_cl = np.full((_batch_size, x_clustered[0].shape[0]), -99.)
                
                # padding the vectors with respect to the maximum sequence of each batch and not the whole training data
                if with_clustering and only_clustered==False:
                    #if len(x_clustered[0].shape) == 2:
                    #    x_cl = np.full((_batch_size, x_clustered[0].shape[0], x_clustered[0].shape[1]), -99.)
                    #elif len(x_clustered[0].shape) == 1:
                    #    x_cl = np.full((_batch_size, x_clustered[0].shape[0]), -99.)
                        
                    for i in range(_batch_size):
                        li = b * batch_size + i
                        x_[i, 0:len(x_list[li]), :] = x_list[li]
                        y_[i] = y_list[li]
                        x_cl[i] = x_clustered[li]
                    
                    yield [x_,x_cl], y_ 
                        
                elif only_clustered:
                    for i in range(_batch_size):
                        li = b * batch_size + i
                        y_[i] = y_list[li]
                        x_cl[i] = x_clustered[li]
                    
                    yield x_cl, y_                
                
                else:
                    for i in range(_batch_size):
                        li = b * batch_size + i
                        x_[i, 0:len(x_list[li]), :] = x_list[li]
                        y_[i] = y_list[li]
                    
                    yield x_, y_                    



            