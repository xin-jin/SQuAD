from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib import rnn
# from tensorflow.nn.rnn_cell import BasicLSTMCell

from evaluate import exact_match_score, f1_score

import qa_util
import random
import os
from qa_decoder_lib import EncoderCoattention, DecoderDynamic

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class Encoder(object):
    def __init__(self, config):
        self.config = config
        self.xavier_initializer = tf.contrib.layers.xavier_initializer()

    def _get_LSTMCell(self, dropout):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(
            self.config.hidden_size,
            forget_bias=1.0,
            initializer=self.xavier_initializer)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
            lstm_cell, output_keep_prob=dropout)
        return lstm_cell

    def _encode_LSTM(self, inputs, seq_lens, dropout, scope):
        lstm_cell = self._get_LSTMCell(dropout)
        outputs, (c_T, h_T) = tf.nn.dynamic_rnn(
            lstm_cell,
            inputs,
            sequence_length=seq_lens,
            dtype=tf.float32,
            scope=scope)
        return outputs, h_T

    def _encode_biLSTM(self, inputs, seq_lens, dropout, scope):
        forward_cell = self._get_LSTMCell(dropout)
        backward_cell = self._get_LSTMCell(dropout)
        outputs, state = tf.nn.bidirectional_dynamic_rnn(
            forward_cell,
            backward_cell,
            inputs,
            sequence_length=seq_lens,
            dtype=tf.float32,
            scope=scope)
        return tf.concat(2, outputs), state

    def encode(self, d_embeds, q_embeds, d_lens, q_lens, dropout):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """

        d_states, _ = self._encode_biLSTM(
            d_embeds, d_lens, dropout, scope='context')
        # _, q_state = tf.tanh(
        #     self._encode_LSTM(
        #         q_embeds, q_lens, scope='question'))
        _, q_state = self._encode_LSTM(
            q_embeds, q_lens, dropout, scope='question')
        # q_state = tf.tanh(q_state)
        return d_states, q_state

    # def create_coattention_inputs(self, d_states, q_states, m, n):
    #     """
    #     d_states has shape [batch_size, m+1, hidden_size],
    #     q_states has shape [batch_size, n+1, hidden_size],
    #     where m is the max length of the sequence in the batch,
    #     similar for n
    #     """
    #     d_states_t = tf.transpose(d_states, [0, 2, 1])
    #     d_states_t_reshape = tf.reshape(d_states_t, [-1, m+1])
    #     q_states_reshape = tf.reshape(q_states, [-1, self.config.hidden_size])
    #     affinity = tf.matmul(d_states_t_reshape, q_states_reshape)

    #     return 0

    # def encode_coattention(self, d_embeds, q_embeds, pad_info):
    #     d_states = encode_LSTM(d_embeds, pad_info.cseq_lens)
    #     q_states = tf.tanh(encode_LSTM(q_embeds, pad_info.qseq_lens))
    #     # append the sentinel to the end of each sequence
    #     for i in range(self.config.batch_size):
    #         tf.scatter_update(d_states, [i, pad_info.cseq_lens[i]], sentinel)
    #         tf.scatter_update(q_states, [i, pad_info.qseq_lens[i]], sentinel)


class Decoder(object):
    def __init__(self, config):
        def initialize_attention(scope):
            xavier_initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope(scope):
                tf.get_variable(
                    'W1',
                    shape=[
                        self.config.hidden_size * 2, self.config.hidden_size
                    ],
                    initializer=xavier_initializer)
                tf.get_variable(
                    'W2',
                    shape=[self.config.hidden_size, self.config.hidden_size],
                    initializer=xavier_initializer)
                tf.get_variable(
                    'v',
                    shape=[self.config.hidden_size],
                    initializer=tf.zeros_initializer)

        # self.output_size = output_size
        self.config = config
        self.xavier_initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('qa'):
            initialize_attention('begin')
            initialize_attention('end')

    def _decode_LSTM(self, inputs, seq_lens, init, dropout, scope):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(
            self.config.hidden_size,
            forget_bias=1.0,
            initializer=self.xavier_initializer)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
            lstm_cell, output_keep_prob=dropout)
        outputs, (c_T, h_T) = tf.nn.dynamic_rnn(
            lstm_cell,
            inputs,
            sequence_length=seq_lens,
            dtype=tf.float32,
            initial_state=init,
            scope=scope)
        return outputs, h_T

    def decode(self, encoder_outputs, d_lens, q_lens, dropout):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """

        d_states, q_state = encoder_outputs

        def pointer_preds(encoder_states, decoder_state, scope):
            with tf.variable_scope(scope, reuse=True):
                W1 = tf.get_variable('W1')
                W2 = tf.get_variable('W2')
                v = tf.reshape(
                    tf.get_variable('v'), [self.config.hidden_size, 1])
                max_t = tf.shape(encoder_states)[1]
                Wq = tf.matmul(decoder_state, W2)
                encoder_states = tf.reshape(encoder_states,
                                            [-1, self.config.hidden_size * 2])
                Wd = tf.matmul(encoder_states, W1)
                u = tf.matmul(tf.tanh(Wd + tf.tile(Wq, [max_t, 1])), v)
                u = tf.reshape(u, [-1, max_t])

                # u = tf.matmul(
                #     tf.tanh(Wd + tf.reshape(
                #         Wq, [self.config.batch_size, max_t, 1])),
                #     tf.reshape(v, [self.config.hidden_size, 1]))
                # u = tf.reshape(u, [self.batch_size, max_t])
            return u

        # _, begin_state = self._decode_LSTM(d_states, d_lens, q_state, dropout,
        #                                    'a_s')
        # _, end_state = self._decode_LSTM(d_states, d_lens, q_state, dropout,
        #                                  'a_e')
        begin_preds = pointer_preds(d_states, q_state, 'begin')
        end_preds = pointer_preds(d_states, q_state, 'end')

        return begin_preds, end_preds


class QASystem(object):
    def __init__(self, encoder, decoder, config, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        self.config = config
        self.encoder = encoder
        self.decoder = decoder

        # ==== set up placeholder tokens ========
        self.questions_placeholder = tf.placeholder(
            shape=[None, None], dtype=tf.int32, name='questions')
        self.questions_len_placeholder = tf.placeholder(
            shape=[None], dtype=tf.int32, name='questions_len')
        self.contexts_placeholder = tf.placeholder(
            shape=[None, None], dtype=tf.int32, name='contexts')
        self.contexts_len_placeholder = tf.placeholder(
            shape=[None], dtype=tf.int32, name='contexts_len')
        self.begin_placeholder = tf.placeholder(
            shape=[None], dtype=tf.int32, name='begin')
        self.end_placeholder = tf.placeholder(
            shape=[None], dtype=tf.int32, name='end')
        self.dropout_placeholder = tf.placeholder(
            shape=[], dtype=tf.float32, name='dropout')

        # ==== assemble pieces ====
        with tf.variable_scope(
                "qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()
            self.setup_optimizer()

        # ==== set up training/updating procedure ====
        self.saver = tf.train.Saver()

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        encoder_outputs = self.encoder.encode(
            self.d_embeds, self.q_embeds, self.contexts_len_placeholder,
            self.questions_len_placeholder, self.dropout_placeholder)
        self.begin_preds, self.end_preds = self.decoder.decode(
            encoder_outputs, self.contexts_len_placeholder,
            self.questions_len_placeholder, self.dropout_placeholder)

    def setup_optimizer(self):
        # self.train_op = tf.train.AdamOptimizer(self.config.lr).minimize(
        #     self.loss)
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        grads, vars = zip(*optimizer.compute_gradients(self.loss))
        grads, vars = list(grads), list(vars)
        if self.config.clip_gradients:
            grads, self.grad_norm = tf.clip_by_global_norm(
                grads, self.config.max_grad_norm)
        else:
            self.grad_norm = tf.global_norm(grads)
        self.train_op = optimizer.apply_gradients(zip(grads, vars))

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                self.begin_preds, self.begin_placeholder)) + tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        self.end_preds, self.end_placeholder))

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        # with vs.variable_scope("embeddings"):
        #     pass
        embeddings = qa_util.load_embeddings(self.config.embed_path)
        if self.config.constant_embeddings:
            self.embeddings = tf.constant(embeddings, dtype=tf.float32)
        else:
            self.embeddings = tf.Variable(embeddings, dtype=tf.float32)
        self.d_embeds = tf.nn.embedding_lookup(self.embeddings,
                                               self.contexts_placeholder)
        self.q_embeds = tf.nn.embedding_lookup(self.embeddings,
                                               self.questions_placeholder)

    def _create_feed_dict(self,
                          contexts_batch,
                          contexts_len_batch,
                          questions_batch,
                          questions_len_batch,
                          begin_batch,
                          end_batch,
                          dropout=1.):
        feed_dict = {}
        feed_dict[self.contexts_placeholder] = contexts_batch
        feed_dict[self.contexts_len_placeholder] = contexts_len_batch
        feed_dict[self.questions_placeholder] = questions_batch
        feed_dict[self.questions_len_placeholder] = questions_len_batch
        feed_dict[self.begin_placeholder] = begin_batch
        feed_dict[self.end_placeholder] = end_batch
        feed_dict[self.dropout_placeholder] = dropout
        return feed_dict

    def optimize(self, session, input_feed):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        output_feed = [self.train_op, self.loss]

        outputs = session.run(output_feed, feed_dict=input_feed)

        return outputs

    def test(self, session, contexts, questions, begins, ends, pad_info):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = self._create_feed_dict(contexts, pad_info.cseq_lens,
                                            questions, pad_info.qseq_lens,
                                            begins, ends)

        output_feed = self.loss

        outputs = session.run(output_feed, input_feed)

        return outputs

    def _create_decode_feed_dict(self, context, question):
        feed_dict = {}
        feed_dict[self.contexts_placeholder] = [context]
        feed_dict[self.questions_placeholder] = [question]
        feed_dict[self.contexts_len_placeholder] = [len(context)]
        feed_dict[self.questions_len_placeholder] = [len(question)]
        feed_dict[self.dropout_placeholder] = 1.
        return feed_dict

    def decode(self, session, context, question):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = self._create_decode_feed_dict(context, question)

        output_feed = [self.begin_preds, self.end_preds]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, context, question):
        """
        context and question are for only one example
        """

        yp, yp2 = self.decode(session, context, question)

        # a_s = np.argmax(yp, axis=1)
        # a_e = np.argmax(yp2, axis=1)
        a_s = np.argmax(yp)
        a_e = np.argmax(yp2)

        return (a_s, a_e)

    def validate(self, sess, valid_batch):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0.
        count = 0

        # for context, question, begin, end in valid_dataset:
        #     valid_cost = self.test(sess, context, question, begin, end)
        for batch, pad_info in valid_batch:
            valid_cost += self.test(sess, batch[0], batch[1], batch[2],
                                    batch[3], pad_info)
            count += 1

        return valid_cost / count

    def evaluate_answer(self, session, dataset, sample=100, log=False):
        """
        Our dataset format: a list of (context, question, begin, end)


        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        if len(dataset) > sample:
            dataset = random.sample(dataset, sample)

        f1, em = 0., 0.
        for context, question, begin, end in dataset:
            a_s, a_e = self.answer(session, context, question)
            a_s = min(a_s, len(context) - 1)
            a_e = min(a_e, len(context) - 1)
            if a_s > a_e:
                a_s, a_e = a_e, a_s
            prediction = context[a_s:(a_e + 1)]
            prediction = ' '.join([str(x) for x in prediction])
            ground_truth = context[begin:(end + 1)]
            ground_truth = ' '.join([str(x) for x in ground_truth])
            f1 += f1_score(prediction, ground_truth)
            em += exact_match_score(prediction, ground_truth)

        f1 = f1 * 100 / len(dataset)
        em = em * 100 / len(dataset)

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em,
                                                                 sample))

        return f1, em

    def _train_on_batch(self, session, contexts_batch, questions_batch,
                        begin_batch, end_batch, pad_info):
        input_feed = self._create_feed_dict(
            contexts_batch, pad_info.cseq_lens, questions_batch,
            pad_info.qseq_lens, begin_batch, end_batch, self.config.dropout)
        _, loss = self.optimize(session, input_feed)
        return loss

    def _run_epoch(self, session, train_data):
        loss = 0.
        count = 0
        for i, (batch, pad_info) in enumerate(
                qa_util.pad_minibatched(train_data, self.config.batch_size)):
            loss += self._train_on_batch(session, batch[0], batch[1], batch[2],
                                         batch[3], pad_info)
            count += 1
            if i == 0:
                print('initial loss: {:f}'.format(loss))
            if (i + 1) % self.config.print_after_batchs == 0:
                print('training loss after {:d} batches is {:f}'.format(
                    i + 1, loss / count))
        return loss / count

    def train(self, session, save_train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(
            map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" %
                     (num_params, toc - tic))

        train_data = qa_util.read_train_val_data(self.config.train_path,
                                                 'train')
        val_data = qa_util.read_train_val_data(self.config.val_path, 'val')
        val_batch = list(
            qa_util.pad_minibatched(val_data, self.config.batch_size))
        best_f1 = 0.
        for epoch in range(self.config.n_epochs):
            print('Begin epoch {:d}'.format(epoch + 1))
            train_cost = self._run_epoch(session, train_data)
            print('Finished epoch {:d}'.format(epoch + 1))

            val_cost = self.validate(session, val_batch)
            print('training loss: {:f}, validation loss: {:f}'.format(
                train_cost, val_cost))

            f1, em = self.evaluate_answer(session, val_data)
            print('dev_f1: {:f}, dev_em: {:f}'.format(f1, em))
            if f1 > best_f1:
                self.saver.save(session,
                                os.path.join(save_train_dir,
                                             self.config.model_output))
                print("Model saved in file: %s" %
                      os.path.join(save_train_dir, self.config.model_output))

