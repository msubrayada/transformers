from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization
import tensorflow as tf

"""
Base Attention class
"""
class BaseAttention(Layer):
    def __init__(self):
        super(BaseAttention, self).__init__()
    
    def get_config(self):
        config = super(BaseAttention, self).get_config()        
        return config
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, key_dim).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, key_dim)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.key_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def get_attention(self, q, v, k=None, attention_mask=None):
        return self.call(q, v, k, attention_mask, return_attention_scores = True)[1]
    
    
"""
X-Linear MultiHead Attention
"""
class X_Linear_Attention(BaseAttention):
    def __init__(self, num_heads, key_dim):
        super(X_Linear_Attention, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.depth = key_dim * num_heads

        self.wkq = Dense(self.depth)
        self.wvq = Dense(self.depth)
        self.wk = Dense(self.depth)
        self.wv = Dense(self.depth)
        
        self.wKB = Dense(512)
        self.wb = Dense(1)
        self.we = Dense(self.key_dim)

        self.dense = Dense(self.key_dim)

    def get_config(self):
        config = super(X_Linear_Attention, self).get_config()
        config.update({"num_heads":self.num_heads, "key_dim":self.key_dim})
        return config
    
    
    
    def product_attention(self, x, y):        
        return tf.einsum("...QD, ...KD -> ...QKD", tf.nn.elu(x), tf.nn.elu(y))
    
    
    def x_linear_attention(self, qk, qv, k, v, mask):
        Bk = self.product_attention(qk, k) # (bs, H, l_q, l_k, key_dim)
        BK = tf.nn.relu(self.wKB(Bk))  # (bs, H, l_q, l_k, 512)
        attention_logits = self.wb(BK) # (bs, H, l_q, l_k, 1)
        
        Bv = self.product_attention(qv, v) # (bs, H, l_q,, l_k, key_dim)
        
        if mask is not None:
            mask = tf.cast(mask, tf.float32) # (bs, l_q, l_k)
            mask_ext = tf.expand_dims(mask, 1)
            mask_ext = tf.expand_dims(mask_ext, -1) # (bs, 1, l_q, l_k, 1)
            attention_logits += ((1 - mask_ext) * -1e9)            
            
            BK_pool = tf.reduce_sum(BK * mask_ext, -2) / tf.reduce_sum(mask_ext, -2)
        else:
            BK_pool = tf.reduce_mean(BK, -2) # (bs, H, l_q, 512)
        
        
        attention_weights = tf.nn.softmax(attention_logits, axis=-2) # (bs, H, l_q, l_k, 1)
        
        attention_channel = tf.math.sigmoid(self.we(BK_pool)) # (bs, H, l_q, key_dim)
        
        output = tf.reduce_sum(Bv * attention_weights, axis=-2) # (bs, H, l_q, key_dim)
        
        output = output * attention_channel
        return output, attention_weights
        
    
    def call(self, q, v, k=None, attention_mask=None, return_attention_scores=False):
        if (k == None):
            k = v
        batch_size = tf.shape(q)[0]

        qk = self.wkq(q)  # (batch_size, seq_len_q, depth)
        qv = self.wvq(q)  # (batch_size, seq_len_q, depth)
        k = self.wk(k)  # (batch_size, seq_len_v, depth)
        v = self.wv(v)  # (batch_size, seq_len_v, depth)

        qk = self.split_heads(qk, batch_size)  # (batch_size, num_heads, seq_len_q, key_dim)
        qv = self.split_heads(qv, batch_size)  # (batch_size, num_heads, seq_len_q, key_dim)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_v, key_dim)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, key_dim)

        # attention.shape == (batch_size, num_heads, seq_len_q, key_dim)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        attention, attention_weights = self.x_linear_attention(qk, qv, k, v, attention_mask)        

        attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, key_dim)
        concat_attention = tf.reshape(attention, (batch_size, -1, self.depth))  # (batch_size, seq_len_q, depth)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, key_dim)
        if (return_attention_scores == True):
            return output, attention_weights
        return output
    
    


    
"""
MultiHead Dot product attention
"""
class MHAttention(BaseAttention):
    def __init__(self, num_heads, key_dim):
        super(MHAttention, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.depth = key_dim * num_heads

        self.wq = Dense(self.depth)
        self.wk = Dense(self.depth)
        self.wv = Dense(self.depth)

        self.dense = Dense(self.key_dim)

    def get_config(self):
        config = super(MHAttention, self).get_config()
        config.update({"num_heads":self.num_heads, "key_dim":self.key_dim})
        return config
    
    
    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True) 
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:                        
            mask = tf.cast(mask, tf.float32) # (bs, l_q, l_k)
            mask_ext = tf.expand_dims(mask, 1)
            #mask_ext = tf.expand_dims(mask_ext, -1) # (bs, 1, l_q, l_k, 1)            
            scaled_attention_logits += ((1 - mask_ext) * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  
        output = tf.matmul(attention_weights, v) 
        return output, attention_weights
    
    def call(self, q, v, k=None, attention_mask=None, return_attention_scores=False):
        if (k == None):
            k = v
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, depth)
        k = self.wk(k)  # (batch_size, seq_len, depth)
        v = self.wv(v)  # (batch_size, seq_len, depth)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, key_dim)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, key_dim)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, key_dim)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, key_dim)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, attention_mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, key_dim)

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.depth))  # (batch_size, seq_len_q, depth)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, key_dim)
        if (return_attention_scores == True):
            return output, attention_weights
        return output
    
"""
MultiHead Meshed Memory attention
"""
class EncoderMMAttention(MHAttention):
    def __init__(self, num_heads, key_dim, memory_vectors=40):
        super(EncoderMMAttention, self).__init__(num_heads, key_dim)
        self.num_heads = num_heads
        self.key_dim = key_dim + memory_vectors
        self.memory_vectors = memory_vectors

        self.depth = (key_dim + memory_vectors) * num_heads

        self.wq = Dense(self.depth)
        self.wk = Dense(self.depth)
        self.wv = Dense(self.depth)

        self.dense = Dense(key_dim)

    def get_config(self):
        config = super(EncoderMMAttention, self).get_config()
        config.update({"memory_vectors": self.memory_vectors})
        return config
    
class DecoderMMAttention(MHAttention):
    def __init__(self, num_heads, key_dim, num_encoders=6):
        super(DecoderMMAttention, self).__init__(num_heads, key_dim)        
        self.N = num_encoders
        self.dense_alpha = [Dense(key_dim) for i in range(self.N)]    
        
    def call(self, q, v, k=None, attention_mask=None, return_attention_scores=False):        
        assert self.N == len(v)
        
        C = []        
        for i in range(self.N):
            c, att_scores = super(DecoderMMAttention, self).call(q, v[i], k=None, attention_mask=None, return_attention_scores=True)
            C.append(c)
            
        C = tf.cast(C, tf.float32) # (N, bs, n_q, d)
        C = tf.transpose(C, [1,0,2,3]) # (bs, N, n_q, d)
                        
        # q.shape (bs, n_q, d)
        q = tf.expand_dims(q, 1)  # (bs, 1, n_q, d)
        q = tf.tile(q, [1,self.N,1,1]) # (bs, N, n_q, d)
        
        x = tf.concat([q, C], axis=3)  # (bs, N, n_q, 2*d)
        stack = tf.unstack(x, axis=1)  # list of len N with shapes (bs, n_q, 2*d)
        alphas = []
        for e, D in zip(stack, self.dense_alpha):
            alphas.append(tf.math.sigmoid(D(e)))
        
        alphas = tf.cast(alphas, tf.float32)
        alphas = tf.transpose(alphas, [1,0,2,3])
        
        output = alphas * C
        output = tf.reduce_sum(output, axis=1)
        if (return_attention_scores == True):
            return output, att_scores
        return output

        

    
    