import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Constant


class PositionalEmbeddingLayer(Layer):
    def __init__(self,seqlen,embedding_dim,**kwargs):
        super().__init__(**kwargs)
        self.seqlen=seqlen
        self.embedding_dim=embedding_dim

        self.positional_matrix_generation()

        self.positional_embedding=self.add_variable(
            shape=(self.seqlen,self.embedding_dim),
            dtype=tf.float32,
            trainable=True,
            initializer=Constant(self.positional_matrix),
            name="positional_embedding"
        )


    def positional_matrix_generation(self):
        matrix=np.zeros(shape=(self.seqlen,self.embedding_dim))
        n=1000

        for pos in range(self.seqlen):
            for i in range(int(self.embedding_dim/2)):
                denom=np.power(n,(2*i/self.embedding_dim))
                matrix[pos,2*i]=np.sin(pos/denom)
                matrix[pos,2*i+1]=np.cos(pos/denom)

        self.positional_matrix=matrix


    def call(self,x):
        batch_size=x.shape[0]
        embedding=tf.broadcast_to(self.positional_embedding,(batch_size,self.seqlen,self.embedding_dim))
        embedding=tf.transpose(embedding,perm=[0,2,1])

        out=tf.math.add(x,embedding)
        return out
        

    def get_config(self):
        config=super().config()
        config.update({
            "seqlen":self.seqlen,
            "embedding_dim":self.embedding_dim
        })

        return config
        

    def compute_output_shape(self,input_shape):
        return input_shape
        
        