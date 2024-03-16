import tensorflow as tf
import numpy as np
class ScaledDotProductAttention(tf.keras.layers.Layer):
  def __init__(self,):
    super().__init__()

  def call(self,Q,K,V,mask = None): #BaseSelfAttention
      """
      Calculate attention weights
      
      Arguments:
          Q = Query, shape = (...,Sequence length = Tq, dimension = dq) 
          K = Query, shape = (...,Sequence length = Tk, dimension = dk) 
          dq = dk
          V = Query, shape = (...,Sequence length = Tv, dimension = dv) 
          Tk = Tv
          mask = Q*K(Transposed) elements to be excluded from attention calculation, shape = (Tq,Tk), default = None
      Returns:
          Attention scores of shape ()
      """
      #compute dot product
      QK_t = tf.matmul(Q,K, transpose_b=True) #gives shape (..., Tq, Tk)

      dk = K.shape[-1] #gets embedding dimension for the key, to be used in scaled softmax calculation
      scaling_QK_t = QK_t/np.sqrt(dk)
      if mask!=None:
          scaling_QK_t += (1.0 - mask)*(-1e9)#inverting attention masks, not used in actual paper, find out why invert? maybe for cross attention?
      softmax_QK_t = tf.nn.softmax(scaling_QK_t, axis=-1) #scaled softmax calculation, calculated on the last axis i.e. -1

      attention_output = tf.matmul(softmax_QK_t, V) #shape = (..., Tq, dv)

      return attention_output, softmax_QK_t


## DEBUG scaled_dot_product_attention()

# batch_size,Tq, Tv, dk, dv= 16,9,9,64,128 # we need Tq=Tv
# Q= tf.random.uniform((batch_size,Tq, dk))
# K= tf.random.uniform((batch_size,Tv, dk))
# V= tf.random.uniform((batch_size,Tv, dv))

# A,_=scaled_dot_product_attention(Q, K, V)
# print(A.shape)

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, batch_size):
        """
        Arguments:
            H = number of heads, (=8 given in paper)
            d_model = embedding dimension (=512 in paper)
        Return:
            
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.size_of_head = int(self.d_model/self.num_heads)
        
        # shape of these matrix (batch_size, sequence length, model dimension or d_model) after q,k,v of shape(batch size, sequence length, embedding dimension) is passed to them
        self.Wq = tf.keras.layers.Dense(d_model)#,trainable=True
        self.Wk = tf.keras.layers.Dense(d_model)#,trainable=True
        self.Wv = tf.keras.layers.Dense(d_model)#,trainable=True
        self.WO = tf.keras.layers.Dense(d_model)#,trainable=True
        self.attention = ScaledDotProductAttention()
        self.batch_size = batch_size
        
    def expand(self, value):
        """Expanding dimension from (batch size, sequence length, embedding dimension) to
            (batch_size, num_heads, sequence length, size_of_head) to be used for self attention on the last two axis"""
        value = tf.reshape(value, (self.batch_size, -1, self.num_heads, self.size_of_head)) #Changing to (batch_size, sequence length, num_heads, size_of_head)
        return tf.transpose(value,perm=[0,2,1,3]) #Changing to (batch_size, num_heads, sequence length, size_of_head)

    def call(self, Q, K, V, mask = None):
        
        #batch_size = Q.shape()[0]
        # Q, K, V has shape (batch size, sequence length, embedding dimension)
        query = self.Wq(Q)
        key = self.Wq(K)
        value = self.Wq(V)

        query = self.expand(query)
        key = self.expand(key)
        value = self.expand(value)

        att_output, att_weights = self.attention(query, key, value)

        att_output = tf.transpose(att_output, perm = [0,2,1,3]) #Changing back to (batch_size, sequence length, num_heads, size_of_head)
        att_output = tf.reshape(att_output, (self.batch_size, -1, self.d_model)) #Changing back to (batch_size, sequence length, model dimension or d_model)
        output = self.WO(att_output) #output shape (batch_size, sequence length, model dimension or d_model)

        return output

class point_wise_ffn(tf.keras.layers.Layer):
    def __init__(self, dff, d_model):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
        self.dense2 = tf.keras.layers.Dense(d_model)
    def call(self, att_output):
        dense1 = self.dense1(att_output)
        dense2 = self.dense2(dense1)

        return dense2

"""
def point_wise_FNN(dff, d_model):
    return tf.keras.Sequential([tf.keras.layers.Dense(dff, activation='relu'),
    tf.keras.layers.Dense(d_model)])
"""

def angle_defn(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  # create the sinusoidal pattern for the positional encoding
  angle_rads = angle_defn(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
  
  sines = np.sin(angle_rads[:, 0::2])
  cosines = np.cos(angle_rads[:, 1::2])
  
  pos_encoding = tf.cast(np.concatenate([sines, cosines], axis=-1)[np.newaxis, ...], dtype=tf.float32)
  return pos_encoding 

def positional_encoding__(positions, d_model):
  """
  Precomputes a matrix with all the positional encodings 
  
  Arguments:
      positions (int) -- Maximum number of positions to be encoded 
      d_model (int) -- Encoding size d_model
  
      arguments de get_angles:
          pos -- Column vector containing the positions [[0], [1], ...,[N-1]]
          k --   Row vector containing the dimension span [[0, 1, 2, ..., d-1]]
          d(integer) -- Encoding size
          
  Returns:
      pos_encoding -- (1, position, d_model) A matrix with the positional encodings
  """
  # initialize a matrix angle_rads of all the angles
  pos=np.arange(positions)[:, np.newaxis] #Column vector containing the position span [0,1,..., positions]
  k= np.arange(d_model)[np.newaxis, :]  #Row vector containing the dimension span [[0, 1, ..., d-1]]
  i = k//2
  angle_rads = pos/(10000**(2*i/d_model)) #Matrix of angles indexed by (pos,i)
  
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  #adds batch axis
  pos_encoding = angle_rads[np.newaxis, ...] 
  
  return tf.cast(pos_encoding, dtype=tf.float32)

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, dff, d_model, num_heads, batch_size = 16, rate = 0.1):
        super().__init__()
        """
        Elements in encoded layer
            Layer made up of Multi head attention -> add and normalize -> Point wise feed forward -> add and normalize

        """
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.ffn = point_wise_ffn(dff, d_model)
        self.multiheadattention = MultiHeadSelfAttention(num_heads, d_model, batch_size)

        self.dropout_mha = tf.keras.layers.Dropout(rate)
        self.dropout_ffn = tf.keras.layers.Dropout(rate)
    def call(self, X, mask=None):#trainable=False
        
        #X_norm = self.layernorm1(X)
        attention_output = self.multiheadattention(X, X, X, mask = mask)
        dropout_mha = self.dropout_mha(attention_output)#, trainable)
        add_and_norm_mha = self.layernorm1(dropout_mha + X)

        ffn_output = self.ffn(add_and_norm_mha)
        dropout_ffn = self.dropout_ffn(ffn_output)#, trainable)
        add_and_norm_ffn = self.layernorm2(dropout_ffn+add_and_norm_mha)

        return add_and_norm_ffn #shape (batch_size, sequence length, d_model)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, dff, d_model, num_heads, batch_size = 16, rate = 0.1):
        super().__init__()

        self.layers = [EncoderLayer(dff, d_model, num_heads, batch_size, rate)
                for layer in range(num_layers)]

    def call(self, X, mask=None):# trainable=False,
        #layer_out = [] , for keeping the output for all layers
        for layer in self.layers:
            X = layer(X, mask=mask)# trainable,
        return X

