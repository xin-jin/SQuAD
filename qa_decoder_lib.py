#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic decoder from Dynamic Coattention Network
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys

import tensorflow as tf
import numpy as np

class Config1(object):
    feature_size = 100
    hidden_size = 100
    batch_size = 32
    n_epochs = 10
    lr = 0.001

    def __init__(self, embed_path):
        self.embed_path = embed_path

class HMN(object):
    """ Implementing the highway maxout network as described in the paper"""
    def __init__(self,config,**kwargs):
        """ Initializes the highway network:
            Inputs: config - config object
            Optional: state_size (def 100) - size of hidden layer
                      pool_size (def 16) - number of max poolout layers
                      dropout (def 1) - retain rate in dropout
                      scope (def 'HMN') - scope of variables in tensorflow
        """
        self.state_size = config.hidden_size
        self.u_s = kwargs.get('u_s',config.hidden_size*2)
        self.pool_size = kwargs.get('pool_size',16)
        self.scope = kwargs.get('scope',type(self).__name__)
                                 
        
    def run(self,h_inp,u_start,u_end,U,bool_mask,dropout=1):
        """ Runs the highway network once:
            Inputs: h - initial state (bat x self.state_size)
                    u_start - starting embedding normalized by probability distribution of starting word position (bat x m)
                    u_end - ending embedding by probability distribution of ending word position (bat x m)
                    U - knowledge representation (size bat x m x u_s)
                    bool_mask = mask on d_lens (bat x m)
                    dropout - keep probability in dropout
            Outputs: alpha - logits of the probability distribution of starting/ending position
                     u_alpha - embedding normalized by probability of alpha
                     
        """
        h = tf.concat(1,h_inp)
        self.dropout = dropout
        with tf.variable_scope(self.scope):
            xavier = tf.contrib.layers.xavier_initializer()
            bat,m,_ = tf.unpack(tf.shape(U))

            WD = tf.get_variable(name='WD',initializer=xavier,dtype=tf.float32,
                                 shape = (2*self.state_size+self.u_s*2,self.state_size))
            #Generating embedding of starting and ending position, embedding r of h,p_start,p_end
            
            r = tf.nn.dropout(tf.tanh(tf.matmul(tf.concat(1,(h,u_start,u_end)),WD)),self.dropout) # bat x l
            r = tf.tile(tf.reshape(r,[bat,1,self.state_size]),[1,m,1]) # bat x m x l
            
            # First maxout network
            m1_inp = tf.concat(2,(U,r)) # bat x m x (u_s+l)
            b1 = tf.get_variable(name='b1',initializer=xavier,dtype=tf.float32,
                                 shape=(self.pool_size,self.state_size)) # p x l
            W1 = tf.get_variable(name='W1',initializer=xavier,dtype=tf.float32,
                                 shape=(self.u_s+self.state_size,self.state_size*self.pool_size))
                                # (u_s+l) x l.p
            m1 = tf.matmul(tf.reshape(m1_inp,shape=(-1,self.u_s+self.state_size)),W1) #bat.m x l.p
            m1 = tf.reshape(m1,shape=(-1,self.pool_size,self.state_size)) + b1 # bat.m x p x l
            m1 = tf.nn.dropout(tf.reduce_max(m1,axis=1),self.dropout) #bat.m x l
                       
            # Second maxout network 
            b2 = tf.get_variable(name='b2',initializer=xavier,dtype=tf.float32,
                                 shape=(self.pool_size,self.state_size))
            W2 = tf.get_variable(name='W2',initializer=xavier,dtype=tf.float32,
                                 shape=(self.state_size,self.state_size*self.pool_size))
                                # l x l.p
            m2 = tf.reshape(tf.matmul(m1,W2),[-1,self.pool_size,self.state_size]) + b2 # bat.m x p x l
            m2 = tf.nn.dropout(tf.reduce_max(m2,axis=1),self.dropout) #bat.m x l
            
            # Third maxout 
            alpha_inp = tf.concat(1,[m1,m2]) # bat . m x 2l
            b3 = tf.get_variable(name='b3',initializer=xavier,dtype=tf.float32,
                                 shape=(self.pool_size)) #p
            W3 = tf.get_variable(name='W3',initializer=xavier,dtype=tf.float32,
                                 shape=(2*self.state_size,self.pool_size)) 
                                # 2l x p
            alpha = tf.matmul(alpha_inp,W3) + b3 # bat.m x p
            alpha = tf.reshape(tf.reduce_max(alpha,1),shape=(bat,m)) # bat x m logits 
            alpha = tf.select(bool_mask,alpha,alpha-1000)
            u_alpha = tf.reshape(tf.batch_matmul(tf.reshape(tf.nn.softmax(alpha),[bat,1,m]),U),[bat,self.u_s]) #bat x u_s
            return alpha, u_alpha

class DecoderDynamic(object):
    """Implementing the decoder in the paper
    """
    def __init__(self, config,**kwargs):
        """ Initializing the decoder. 
        Inputs: config - Object of the config class 
        Optional: n_iter - number of iterations (def 4)
                  scope - tf scope (def 'DecodeDynamic')
                  pool_size - parameter to pass downstream to the HMN (def 16)
        """
        
        self.state_size = config.hidden_size
        self.scope = kwargs.get('scope',type(self).__name__)
        self.n_iter = kwargs.get('n_iter',4)
        kwargs_pass = kwargs.copy()
        if 'scope' in kwargs_pass:
            del kwargs_pass['scope']
        with tf.variable_scope(self.scope):
            self.HMNstart = HMN(config,scope='HMNstart',**kwargs_pass)
            self.HMNend = HMN(config,scope='HMNend',**kwargs_pass)
            self.lstmcell = tf.nn.rnn_cell.LSTMCell(self.state_size)
            
    def run_once(self, h, u_start,u_end,U,bool_mask):
        p_start_new,u_start_new = self.HMNstart.run(h,u_start,u_end,U,bool_mask,self.dropout)
        p_end_new, u_end_new = self.HMNend.run(h,u_start_new,u_end,U,bool_mask,self.dropout)
        _,h_new = self.lstmcell(tf.concat(1,(u_start_new,u_end_new)),h,scope=self.scope+'lstm')
        tf.get_variable_scope().reuse_variables()
        return h_new,p_start_new,p_end_new
            
    def decode(self,U,d_lens,q_lens=None,dropout=1):
        self.dropout = dropout
        with tf.variable_scope(self.scope):
            bat,m,_ = tf.unpack(tf.shape(U))
            bool_mask = tf.sequence_mask(d_lens,m)
            h = self.lstmcell.zero_state(bat,dtype=tf.float32)
            d_lens_float = tf.reshape(tf.cast(d_lens,dtype=tf.float32),[-1,1])
            u_end = u_start = tf.div(tf.reduce_sum(U,axis=1),d_lens_float)
            
            for ind in range(self.n_iter):
                h,p_start,p_end = self.run_once(h,u_start,u_end,U,bool_mask)
        return p_start,p_end

class EncoderCoattention(object):
    def __init__(self, config,**kwargs):
        self.state_size = config.hidden_size
        self.scope = kwargs.get('scope',type(self).__name__)
        with tf.variable_scope(self.scope):
            xavier = tf.contrib.layers.xavier_initializer()
            self.qc1 = tf.nn.rnn_cell.LSTMCell(self.state_size,initializer=xavier) #self.size = l
            self.qc2 = tf.nn.rnn_cell.LSTMCell(self.state_size,initializer=xavier)
            self.dc1 = tf.nn.rnn_cell.LSTMCell(self.state_size,initializer=xavier)
            self.dc2 = tf.nn.rnn_cell.LSTMCell(self.state_size,initializer=xavier)
            self.uc1 = tf.nn.rnn_cell.LSTMCell(self.state_size,initializer=xavier)
            self.uc2 = tf.nn.rnn_cell.LSTMCell(self.state_size,initializer=xavier)
            self.d_sentinel = tf.Variable(name='d_sentinel',initial_value=xavier(shape=(1,1,2*self.state_size)),
                                              dtype=tf.float32)
            self.q_sentinel = tf.Variable(name='q_sentinel',initial_value =xavier(shape=(1,1,2*self.state_size)),
                                              dtype=tf.float32)
            

    def encode(self, d_embeds, q_embeds, d_lens,q_lens,dropout=1):
        """
        Runs the encoding based on coattention

        Inputs: d_embeds - Embeddings of the context document (bat x m x l)
                q_embeds - Embeddings of the questions (bat x n x l)
                d_lens - Length of the document (bat)
                q_lens - Length of the question (bat)
        Optional: dropout - keep probability of dropout (def 1)
        Outputs: U - knowledge representation of the question (bat x m x u_s)
        """
        self.dropout = dropout       
        with tf.variable_scope(self.scope):
            Q,final_q = tf.nn.bidirectional_dynamic_rnn(self.qc1,self.qc2,q_embeds,sequence_length=q_lens,dtype=tf.float32,scope='QBiRNN') 
            Q = tf.nn.dropout(tf.concat(2,Q),self.dropout) # Q is batchsize x n x 2l
            bat,_,__ = tf.unpack(tf.shape(Q))
            Q = tf.concat(1,[Q,tf.tile(self.q_sentinel,[bat,1,1])])
            
            D,final_d = tf.nn.bidirectional_dynamic_rnn(self.dc1,self.dc2,d_embeds,sequence_length=d_lens,dtype=tf.float32,scope='DBiRNN')
            D = tf.nn.dropout(tf.concat(2,D),self.dropout) #D is bat x m x 2l
            D = tf.concat(1,[D,tf.tile(self.d_sentinel,[bat,1,1])])
            
            L = tf.batch_matmul(D,Q,adj_y=True) #L is bat x m x n
            AQ = tf.nn.softmax(L) # bat x m x n
            AD = tf.nn.softmax(tf.transpose(L,perm=[0,2,1])) # AD is bat x n x m
            CQ = tf.batch_matmul(AQ,D,adj_x=True) # bat x n x 2l
            CD = tf.batch_matmul(AD,tf.concat(2,[Q,CQ]),adj_x=True) # bat x m x 4l
            
            U,final_u = tf.nn.bidirectional_dynamic_rnn(self.uc1,self.uc2,tf.concat(2,[D,CD]),sequence_length=d_lens,dtype=tf.float32,scope='UBiRNN') 
            U = tf.concat(2,U) #bat x m x 2l
            
        return U

############################# Testing functions #######################################################
def test_encode():
    tf.reset_default_graph()
    with tf.variable_scope('test_encode'):
        
        print(tf.get_variable_scope().reuse)
        conf_obj = Config1('foo')
        enc_obj = EncoderCoattention(conf_obj)
#        enc_obj = Encoder(conf_obj)
        
        
        with tf.Session() as sess:
            d_embeds = tf.random_normal(shape=(2,10,3))
            q_embeds = tf.random_normal(shape=(2,5,3))
            d_lens = tf.cast(tf.ceil(tf.random_uniform(shape=(2,))*10),dtype=tf.int32)
            q_lens = tf.cast(tf.ceil(tf.random_uniform(shape=(2,))*5),dtype=tf.int32)
            enc_embed = enc_obj.encode(d_embeds,q_embeds,d_lens,q_lens)
            init = tf.global_variables_initializer()
            sess.run(init)
            print(sess.run(enc_embed))
            tf.get_variable_scope().reuse_variables()
            d_embeds = tf.random_normal(shape=(2,10,3))
            q_embeds = tf.random_normal(shape=(2,5,3))
            d_lens = tf.cast(tf.ceil(tf.random_uniform(shape=(2,))*10),dtype=tf.int32)
            q_lens = tf.cast(tf.ceil(tf.random_uniform(shape=(2,))*5),dtype=tf.int32)
            enc_embed = enc_obj.encode(d_embeds,q_embeds,d_lens,q_lens)
            print(sess.run(enc_embed))
            
def test_decode():
    tf.reset_default_graph()
    with tf.variable_scope('test_decode'):
        conf_obj = Config1('foo')
        dec_obj = DecoderDynamic(conf_obj)
        U = tf.random_normal(shape=(3,10,conf_obj.hidden_size*2))
        d_lens = tf.cast(tf.ceil(tf.random_uniform(shape=(3,))*10),dtype=tf.int32)
        q_lens = 0
        p_s = dec_obj.decode(U,d_lens,q_lens)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            print(sess.run((p_s)))

def test_HMN():
    tf.reset_default_graph()
    with tf.variable_scope('test_hmn'):
        conf_obj = Config1('foo')
        u_s=5
        m=10
        hmn_obj = HMN(conf_obj,u_s=u_s)
        U = tf.random_normal(shape=(3,m,u_s))
        d_lens = tf.cast(tf.ceil(tf.random_uniform(shape=(3,))*m),dtype=tf.int32)
        bool_mask = tf.sequence_mask(d_lens,m)
        d_lens_float = tf.reshape(tf.cast(d_lens,dtype=tf.float32),[-1,1])
        u_end = u_start = tf.div(tf.reduce_sum(U,axis=1),d_lens_float)
        h = tf.zeros(shape=(3,2*conf_obj.hidden_size),dtype=tf.float32)
        p_s,_ = hmn_obj.run(h,u_start,u_end,U,bool_mask)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            print(sess.run((p_s)))
            
if __name__ == "__main__":
    test_decode()
    
#       def run(self,h_inp,u_start,u_end,U,bool_mask,dropout=1):
#        """ Runs the highway network once:
#            Inputs: h - initial state (bat x self.state_size)
#                    u_start - starting embedding normalized by probability distribution of starting word position (bat x m)
#                    u_end - ending embedding by probability distribution of ending word position (bat x m)
#                    U - knowledge representation (size bat x m x u_s)
#                    bool_mask = mask on d_lens (bat x m)
#                    dropout - keep probability in dropout
#            Outputs: alpha - logits of the probability distribution of starting/ending position
#                     u_alpha - embedding normalized by probability of alpha
#                     
#        """
#        h = tf.concat(1,h_inp)
#        self.dropout = dropout or 1
#        with tf.variable_scope(self.scope):
#            xavier = tf.contrib.layers.xavier_initializer()
#            bat,m,_ = tf.unpack(tf.shape(U))
#
#            WD = tf.get_variable(name='WD',initializer=xavier,dtype=tf.float32,
#                                 shape = (2*self.state_size+self.u_s*2,self.state_size))
#            #Generating embedding of starting and ending position, embedding r of h,p_start,p_end
#            
#            r = tf.nn.dropout(tf.tanh(tf.matmul(tf.concat(1,(h,u_start,u_end)),WD)),self.dropout) # bat x l
#            r = tf.tile(tf.reshape(r,[bat,1,self.state_size]),[1,m,1]) # bat x m x l
#            
#            # First maxout network
#            m1_inp = tf.tile(tf.reshape(tf.concat(2,(U,r)),[bat,m,1,1,-1]),[1,1,self.pool_size,1,1]) # bat x m xp x 1 x (u_s+l)
#            b1 = tf.get_variable(name='b1',initializer=xavier,dtype=tf.float32,
#                                 shape=(self.pool_size,self.state_size)) # p x l
#            W1 = tf.get_variable(name='W1',initializer=xavier,dtype=tf.float32,
#                                 shape=(self.pool_size,self.u_s+self.state_size,self.state_size))
#                                # p x (u_s+l) x l
#            W1_rep = tf.tile(tf.reshape(W1,[1,1,self.pool_size,self.u_s+self.state_size,self.state_size]),[bat,m,1,1,1])
#                                # bat x m x p x u_s+l x l
#            m1 = tf.reshape(tf.batch_matmul(m1_inp,W1_rep),[bat,m,self.pool_size,self.state_size])+b1 # bat x m x p x l
#            m1 = tf.nn.dropout(tf.reduce_max(m1,axis=2),self.dropout) #bat x m x l
#            
#            # Second maxout network 
#            m2_inp = tf.tile(tf.reshape(m1,[bat,m,1,1,self.state_size]),[1,1,self.pool_size,1,1]) # bat x m x p x 1 x l
#            b2 = tf.get_variable(name='b2',initializer=xavier,dtype=tf.float32,
#                                 shape=(self.pool_size,self.state_size))
#            W2 = tf.get_variable(name='W2',initializer=xavier,dtype=tf.float32,
#                                 shape=(self.pool_size,self.state_size,self.state_size))
#                                # p x l x l
#            W2_rep = tf.tile(tf.reshape(W2,[1,1,self.pool_size,self.state_size,self.state_size]),[bat,m,1,1,1])
#                                # bat x m x p x l x l
#            m2 = tf.reshape(tf.batch_matmul(m2_inp,W2_rep),[bat,m,self.pool_size,self.state_size]) + b2 # bat x m x p x l
#            m2 = tf.nn.dropout(tf.reduce_max(m2,axis=2),self.dropout) #bat x m x l
#            
#            m1m2 = tf.concat(2,[m1,m2]) # bat x m x 2l
#            alpha_inp = tf.tile(tf.reshape(m1m2,[bat,m,1,1,2*self.state_size]),[1,1,self.pool_size,1,1]) # bat x m x p x 1 x 2l
#            b3 = tf.get_variable(name='b3',initializer=xavier,dtype=tf.float32,
#                                 shape=(self.pool_size)) #p
#            W3 = tf.get_variable(name='W3',initializer=xavier,dtype=tf.float32,
#                                 shape=(self.pool_size,2*self.state_size,1))
#                                # p x l x 1
#            W3_rep = tf.tile(tf.reshape(W3,[1,1,self.pool_size,2*self.state_size,1]),[bat,m,1,1,1])
#                                # bat x m x p x l x 1
#            alpha = tf.reshape(tf.batch_matmul(alpha_inp,W3_rep),[bat,m,self.pool_size]) + b3 # bat x m x p
#            alpha = tf.reduce_max(alpha,2) # bat x m logits 
#            alpha = tf.select(bool_mask,alpha,alpha-1000)
#            u_alpha = tf.reshape(tf.batch_matmul(tf.reshape(tf.nn.softmax(alpha),[bat,1,m]),U),[bat,self.u_s]) #bat x u_s
#            return alpha, u_alpha
