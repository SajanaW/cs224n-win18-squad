# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


class RNNEncoder(object):

    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.LSTMCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.LSTMCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("ModellingLayer"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class Model_Layer(object):
    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.LSTMCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.LSTMCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out
class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist



class AttentionFlowLayer(object):
    """ Author: Sajana Weerawardhena
        Drawn from Paper: BiDAF
    """

    def __init__(self, keep_prob, l2_lambda):
        """ For the moment, not going to introduce dropout
            regularizing lambdas for the fully connected layers.
            When it will be introduced, init:
             keep_prob,lambda
            """
        self.keep_prob = keep_prob;
        self.l2_lambda = l2_lambda;

    def sim_matrix(self, context_vectors, n_context_vectors, question_vectors, n_question_vectors, batch_size,vector_size):

            """Goal: W_sim^T [c; q; c*q] of shape
            Inputs:
                context_vectors: (type: tensor) Set of Context vectors : (batch_size, n, 2h).
                n_context_vectors: (type: tensor) Number of context vectors.
                question_vectors: (type: tensor) Set of Question vectors: (batch_size,m,2h).
                n_question_vectors: (type: tensor) Number of question vectors.
            Output:
                siml_mat : (type: tensor) Similarity matrix  (batch_size, n_context_vectors, n_question_vectors).
            """

            #naive implementation: not practical because it leads to OOM errors
              #first expand the dimensions of reshape context_vectors:
                # expand_dims context_vector: batch_size,n,2h ->  batch_size,n,1,2h
                # tile : batch_size,n, 1,2h ->  batch_size,n,m,2h
              #Next expand the dimensions of reshape question_vectors:
                # expand_dims question_vectors: batch_size,m,2h ->  batch_size,1,m,2h
                # tile : batch_size,1,m,2h ->  batch_size,n,m,2h
              #Next expand dims, broadcast and concat context annd q_vectors to size batch_size,n,m,6h :
                # expand_dims c1 = context_vector: batch_size,n,2h ->  batch_size,n,1,2h
                # expand_dims c2 = question_vectors: batch_size,m,2h ->  batch_size,1,m,2h
                # broadcast them: c1 * c2 -> batch_size,n,m,2h
              # Concat all of them: batch_size,n,m,6h
              #reshape them into and array of 6h units - that is a unit of question+context+c*q
              # push it through a fully connected layer with 1 hidden unit and no activation
              # so the 6h -> 1 value.
              # reshape the output to shape -1,n_context_vectors, n_question_vectors
            #This was initially implemented and then failed due to OOM Errors.
            # The code for it:
            # """c; of shape: batch_size,n,2h ->  batch_size,n,1,2h"""
            # c = tf.expand_dims(context_vectors, 2)
            # c = tf.tile(c, [1,1,n_question_vectors,1])
            # """c; of shape: batch_size,m,2h ->  batch_size,1,m,2h"""
            # q = tf.expand_dims(question_vectors, 1)
            # q = tf.tile(q, [1,n_context_vectors,1,1])
            # c_mul_q = tf.expand_dims(context_vectors, 2) * tf.expand_dims(question_vectors, 1)
            # """ [c; q; c*q] shape batch_size,n,m,6h  """
            # x = tf.concat([c,q,c_mul_q],3)
            # x = tf.reshape(x,[-1, 3*vector_size])
            # y_cont = tf_layers.fully_connected(x,1, activation_fn=None)
            # y_cont = tf.reshape(y_cont,[-1,n_context_vectors, n_question_vectors]);

            #Smart implementation- burrowed from online implementations involve linearity.
            #context_vectors is of shape: batch_size,n,2h
                    # example:
                    # [ context1:[ context1_word1:[1, 1, 1], context1_word2:[1, 1, 1]]
                    # context2:[ context2_word1:[1, 1, 1], context2_word2:[1, 1, 1]]]
            #reshaped size:  -1, 2h
                    # example:
                    # [ context1_word1:[1, 1, 1], context1_word2:[1, 1, 1]
                    #   context2_word1:[1, 1, 1], context2_word2:[1, 1, 1]]
                    # so just an array of context words.
            input_c = tf.reshape(context_vectors, [-1, vector_size])
            #question_vectors is of shape: batch_size,m,2h
                    # example:
                    # [ q1:[ q1_word1:[1, 1, 1], q1_word2:[1, 1, 1]]
                    # q2:[ q2_word1:[1, 1, 1], q2_word2:[1, 1, 1]]]
            #reshaped size:  -1, 2h
                    # example:
                    # [ q1_word1:[1, 1, 1], q1_word2:[1, 1, 1],
                    #   q2_word1:[1, 1, 1], q2_word2:[1, 1, 1] ]
                    # so just an array of question words.
            input_q = tf.reshape(question_vectors, [-1, vector_size])
            """c; of shape: batch_size,n,2h ->  batch_size,n,1,2h"""
            #Next expand dims, broadcast and concat context annd q_vectors to size batch_size,n,m,6h :
                    # expand_dims c1 = context_vector: batch_size,n,2h ->  batch_size,n,1,2h
                    # expand_dims c2 = question_vectors: batch_size,m,2h ->  batch_size,1,m,2h
                    # broadcast them: c1 * c2 -> batch_size,n,m,2h
            #Then reshape them:
                    #reshape them into and array of 2h units - that is a unit of c*q
                    #For clarity for myself and a naive reader:
                    # when broad casting:
                    # What we have for the context vector after expanded dims and after broadcasting
                    # [ context1:[
                    #    context1_word1:[q1:[1, 1, 1], q2:[same-as-c1w1q1] ... qn:[same-as-c1w1q1]],
                    #    context1_word2:[q1:[distinct-c1w2], q2:[same-as-c1w2q2] ... qn:[same-as-c1w2q2]],
                    #    .....
                    #    context1_wordm:[q1:[distinct-c1wm], q2:[same-as-c1wmqn] ... qn:[same-as-c1wmqn]],
                    #    ]
                    #    ......
                    #    contextm:    ]]
                    # post broadcasting:
                    #   [ context1:[
                    #    context1_word1:[q1:[1, 1, 1], q2:[distinct-q2] ... qn:[distinct-qn]],
                    #    context1_word2: [same-as-context1-word1],
                    #    .....
                    #    context1_wordm:[same-as-context1-word1],
                    #    ]
                    #    ......
                    #    contextm:    ]]
            input_cq = tf.reshape(tf.expand_dims(context_vectors, 2) * tf.expand_dims(question_vectors, 1), [-1, vector_size])

            # TODO: Set this up: Perform dropout on each input.
            input_c = tf.nn.dropout(input_c, self.keep_prob)
            input_q = tf.nn.dropout(input_q, self.keep_prob)
            input_cq = tf.nn.dropout(input_cq, self.keep_prob)

            # For memory efficiency, we compute the linearity piecewise over c, q, and c*q.
            output_c = tf_layers.fully_connected(input_c, 1, activation_fn=None, weights_regularizer=tf_layers.l2_regularizer(scale=self.l2_lambda))

            output_q = tf_layers.fully_connected(input_q, 1, activation_fn=None, weights_regularizer=tf_layers.l2_regularizer(scale=self.l2_lambda))

            output_cq = tf_layers.fully_connected(input_cq, 1, activation_fn=None, weights_regularizer=tf_layers.l2_regularizer(scale=self.l2_lambda))

            # Prepare to add each component together.
            output_c = tf.reshape(output_c, [-1, n_context_vectors, 1])
            output_q = tf.reshape(output_q, [-1, 1, n_question_vectors])
            output_cq = tf.reshape(output_cq, [-1, n_context_vectors, n_question_vectors])

            # then add them up!
            return output_c + output_q + output_cq

    def concat_func(self,context_vectors, c2q_attn, q2c_attn):
        """Apply the beta function for bidaf attention flow
           Goal: concatenate [c, c2q_attn, c * c2q_attn, c * q2c_attn].
        """
        return tf.concat([context_vectors, c2q_attn, context_vectors * c2q_attn, context_vectors * q2c_attn], axis=2)

    def build_graph(self, context_vectors, c_mask,
                        question_vectors, q_mask,scope):
        """Build the BiDAF attention layer component for the compute graph.
        """
        with tf.variable_scope(scope):
            # vector_size = context_vectors.get_shape().as_list()[2]
            batch_size = context_vectors.shape.as_list()[0]
            vector_size = context_vectors.shape.as_list()[2] #2h
            n_context_vectors = context_vectors.shape.as_list()[1]
            n_question_vectors = question_vectors.shape.as_list()[1]
            sim_mat = self.sim_matrix(context_vectors, n_context_vectors, question_vectors, n_question_vectors,batch_size, vector_size)

            #c2q
            # take row wise softmax and then do mat mul
            question_mask_expanded = tf.expand_dims(q_mask, axis=1)
            _, sim_bar = masked_softmax(sim_mat, question_mask_expanded, 2)
            c2q_attn = tf.matmul(sim_bar, question_vectors)

            # q2c
            # comment this out.
            # for each context get the maximum j
            # then take the softmax of these maximums
            # mat mul with n_context_vectors
            context_mask_expanded = tf.expand_dims(c_mask, axis=2)
            #TODO: CHANGE THESE
            _, sim_dbl_bar = masked_softmax(sim_mat, context_mask_expanded, 1)
            sim_dbl_bar = tf.transpose(sim_dbl_bar, (0, 2, 1))
            sim_sim = tf.matmul(sim_bar, sim_dbl_bar)
            q2c_attn = tf.matmul(sim_sim, context_vectors)

            #set up Pretty Print here
            outputs = self.concat_func(context_vectors, c2q_attn, q2c_attn)
        return outputs

""" The base line attention"""
class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output


def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist

class char_cnn(object):
    """ Author: Sajana Weerawardhena
        Drawn from Paper: BiDaf
        Implementing a simple char_cnn for character embedding layer.
    """
    def __init__(self, kernel_size, CNN_filters, stride, char_emb_size, keep_prob):
        """ A simple init"""
        self.kernel_size = kernel_size
        self.CNN_filters = CNN_filters
        self.stride = stride
        self.char_emb_size = char_emb_size
        self.keep_prob = keep_prob

    def build_graph(self, char_embeddings, cq_len, word_len,reuse=False):
        with vs.variable_scope("char_cnn", reuse=reuse):
            char_embeddings = tf.reshape(char_embeddings, [-1, word_len, self.char_emb_size])
            char_embeddings = tf.nn.dropout(char_embeddings, self.keep_prob)
            char_embeddings = tf.layers.conv1d(inputs= char_embeddings, filters=self.CNN_filters,
                                kernel_size=self.kernel_size, activation=tf.nn.relu,
                                strides=self.stride, trainable=True, padding="VALID");
            char_embeddings = tf.reduce_max(char_embeddings, axis=1)
            char_embeddings = tf.reshape(char_embeddings, [-1, cq_len, self.char_emb_size])
        return char_embeddings
