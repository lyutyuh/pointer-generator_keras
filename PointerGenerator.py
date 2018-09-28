# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import sys

sys.path.append('..')
from keras.engine import Layer
import keras.backend as K
from keras.models import Sequential, Model
from keras import activations
from keras import initializers
from keras import optimizers
from keras.initializers import RandomUniform, TruncatedNormal
from keras.layers import Reshape, Embedding, Dot, \
    Input, Dense, Conv2D, CuDNNLSTM, Softmax, Lambda, Bidirectional, Concatenate
from keras.optimizers import Adam
from recurrentshop import LSTMCell
from models.model import BasicModel
from layers.Linear import Linear
import numpy as np
from collections import namedtuple
import tensorflow as tf
from SummarizationUtils.data import Vocab
from SummarizationUtils.batcher import Batcher
from SummarizationUtils.generator import SummarizationGenerator


# Given the reference summary as a sequence of tokens, return the input sequence for the decoder,
# and the target sequence which we will use to calculate loss. 
# The sequence will be truncated if it is longer than max_len. 
# The input sequence must start with the start_id and the target sequence must end with
# the stop_id (but not if it's been truncated).
#
# inp = [start_id] + sequence[:]
# target = sequence[:]
#


class SummarizationModel(BasicModel):
    def __init__(self, vocab, config, hps):
        super(SummarizationModel, self).__init__(config)
        self.__name = 'pointer_generator_summarizer'
        self.config = config
        self.hps = hps

        self.mode = config['mode']
        self.use_coverage = config['use_coverage']
        self.pointer_gen = config['pointer_gen']
        self.embed_trainable = config['train_embed']
        self.embedding_size = config['embed_size']
        self.vsize = config['vocab_size']
        self.rand_unif_init_mag = config['rand_unif_init_mag']
        self.trunc_norm_init_std = config['trunc_norm_init_std']
        self.hidden_units = self.config['hidden_units']
        self.cov_loss_wt = self.config['cov_loss_wt']

        # Initializers:
        self.rand_unif_init = RandomUniform(minval=-self.rand_unif_init_mag,
                                            maxval=self.rand_unif_init_mag,
                                            seed=123)
        self.trunc_norm_init = TruncatedNormal(stddev=self.trunc_norm_init_std)
        # Optimizers:
        self.adg = optimizers.TFOptimizer(
            K.tf.train.AdagradOptimizer(self.hps.lr, initial_accumulator_value=self.hps.adagrad_init_acc))
        # Layers
        self.Emb = Embedding(self.vsize,
                             self.embedding_size,
                             weights=config['embed'],
                             trainable=self.embed_trainable
                             )

        # different dictionary for source and target

        # Bi-directional lstm encoder, return (output, states)
        # Dimension: 2*hidden_units
        # concatenated forward and backward vectors
        self.Encoder = Bidirectional(CuDNNLSTM(self.hidden_units,
                                               return_state=True,
                                               return_sequences=True,
                                               kernel_initializer=self.rand_unif_init
                                               ))
        # Decoder is not bi-directional, perform linear reduction...
        # Dense_layer_dimension=encoder_hidden_units

        # Encoder states and output tensors are separated...
        # to initialize decoder

        # Decoder cell input: [input, state_h, state_c]
        self.DecoderCell = LSTMCell(self.hidden_units,
                                    kernel_initializer=self.rand_unif_init,
                                    bias_initializer="zeros",
                                    recurrent_initializer=self.rand_unif_init)
        # Decoder output projector
        # to probabilities[word_index]
        self.DecoderOutputProjector = Dense(self.vsize,
                                            kernel_initializer=self.trunc_norm_init,
                                            bias_initializer=self.trunc_norm_init,
                                            activation=None
                                            )
        self.ConcatenateAxis1 = Concatenate(axis=1)
        self.ConcatenateLastDim = Concatenate(axis=-1)
        self.StackSecondDim = Lambda(lambda x: K.tf.stack(x, axis=1))
        self.SoftmaxforScore = Softmax(axis=-1)

        self._batch_size = None
        self._enc_batch = None
        self._enc_lens = None
        self._enc_padding_mask = None
        self._enc_batch_extend_vocab = None
        self._max_art_oovs = None
        self._max_art_oovs_inp = None
        self._dec_batch = None
        self._target_batch = None
        self._dec_padding_mask = None
        self._dec_in_state = None
        self._enc_states = None
        self._dec_out_state = None
        self.p_gens = None
        self.prev_coverage = None
        self.coverage = None
        self._coverage_loss = None

        self.check_list = []

        if not self.check():
            pass
        pass

    def ReduceStates(self, concatenated_h, concatenated_c):
        Linear_Reduce_h = Dense(self.config['hidden_units'],
                                activation='relu',
                                kernel_initializer=self.trunc_norm_init,
                                bias_initializer=self.trunc_norm_init
                                )
        Linear_Reduce_c = Dense(self.config['hidden_units'],
                                activation='relu',
                                kernel_initializer=self.trunc_norm_init,
                                bias_initializer=self.trunc_norm_init
                                )
        new_concatenated_h = Linear_Reduce_h(concatenated_h)
        new_concatenated_c = Linear_Reduce_c(concatenated_c)

        return new_concatenated_h, new_concatenated_c

    def TargetEmb(self, target_seq):
        segmented_into_words = Lambda(lambda x: [word for word in K.tf.unstack(x, axis=1)])(target_seq)
        # [(batch_size, w1), (batch_size, w2), ..., (batch_size, w_max_text_length)], length = max_text_length
        embedding_list = [self.Emb(x) for x in segmented_into_words]
        return embedding_list

    def get_encoder_decoder_inputs(self, source_seq, target_seq):
        # Attention:
        emb_enc_inputs = self.Emb(source_seq)  # a tensor
        emb_dec_inputs = self.TargetEmb(target_seq)  # list of embeddings

        ####
        # emb_enc_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch)
        # tensor with shape (batch_size, max_enc_steps, emb_size)
        # emb_dec_inputs = [tf.nn.embedding_lookup(embedding, x) for x in tf.unstack(self._dec_batch, axis=1)]
        # list length max_dec_steps containing shape (batch_size, emb_size)
        ####
        return emb_enc_inputs, emb_dec_inputs

    def attention_decoder(self,
                          decoder_inputs,
                          initial_state,
                          encoder_states,
                          enc_padding_mask,
                          Cell,
                          initial_state_attention=False,
                          pointer_gen=True,
                          use_coverage=False,
                          prev_coverage=None):

        # Requirements:
        # decoder_inputs: A list of 2D Tensors [batch_size x input_size].
        #
        # initial_state: 2D Tensor [batch_size x cell.state_size].
        #                 for the initialization of decoder states
        # encoder_states: (batchsize, timestep, 2*hiddenunits)
        #                 [batch_size, attn_length, attn_size].
        #
        # enc_padding_mask: 2D Tensor [batch_size x attn_length] containing 1s and 0s;
        # indicates which of the encoder locations are padding (0) or a real token (1).
        # cell: rnn_cell.RNNCell defining the cell function and size.
        #
        # initial_state_attention:
        # Note that this attention decoder passes each decoder input through a linear layer
        # with the previous step's context vector to get a modified version of the input.
        # If initial_state_attention is False,
        # on the first decoder step the "previous context vector" is just a zero vector.
        # If initial_state_attention is True, we use initial_state to (re)calculate the previous step's context vector.
        # We set this to False for train/eval mode (because we call attention_decoder once for all decoder steps)
        # and True for decode mode (because we call attention_decoder once for each decoder step).
        #
        # pointer_gen: boolean. If True, calculate the generation probability p_gen for each decoder step.
        #
        # use_coverage: boolean. If True, use coverage mechanism.
        #
        # prev_coverage:
        # If not None, a tensor with shape (batch_size, attn_length). The previous step's coverage vector.
        # This is only not None in decode mode when using coverage.

        # NOTE:
        # To initialize a keras CUDNNLSTM layer's state:
        # ##################################################
        # if isinstance(inputs, list):
        #     initial_state = inputs[1:]
        #     inputs = inputs[0]
        # elif initial_state is not None:
        #     pass
        # elif self.stateful:
        #     initial_state = self.states
        # else:
        #    initial_state = self.get_initial_state(inputs)
        #
        # ##################################################
        attn_size = K.int_shape(encoder_states)[2]
        input_size = K.int_shape(decoder_inputs[0])[1]

        encoder_states = Lambda(lambda x: K.expand_dims(x, axis=2))(encoder_states)
        # now : encoder_states.shape = (batch_size,attn_length,1,attention_vec_size)
        attention_vec_size = attn_size
        W_h_shape = (1, 1, attn_size, attention_vec_size)
        Encoder_Feature_Extractor = Conv2D(kernel_size=(W_h_shape[0], W_h_shape[1]),
                                           filters=W_h_shape[3],
                                           padding="same",
                                           data_format="channels_last"
                                           )
        # W_h = [filter_height, filter_width, in_channels, out_channels]
        encoder_features = Encoder_Feature_Extractor(encoder_states)
        # nn_ops.conv2d(encoder_states, W_h, [1, 1, 1, 1], "SAME")
        # shape (batch_size,attn_length,    1   , attention_vec_size)
        if use_coverage:
            w_c = (1, 1, 1, attention_vec_size)
            Coverage_Feature_Extractor = Conv2D(kernel_size=(w_c[0], w_c[1]),
                                                filters=w_c[3],
                                                padding="same",
                                                data_format="channels_last"
                                                )

        if prev_coverage is not None:
            expand_2_3 = Lambda(lambda x: K.expand_dims(K.expand_dims(x, 2), 3))
            prev_coverage = expand_2_3(prev_coverage)

        # v: shared vector, attention_vec_size-dim -> 1-dim, calculating
        V = Dense(1,
                  use_bias=False,
                  kernel_initializer='glorot_uniform')  # shape : [attention_vec_size]
        Attn_Dist_and_Encoder_States_to_Context_Vector = Lambda(
            lambda X: attn_dist_and_encoder_states_to_context_vector(X, attn_size))
        Masked_Attention = Lambda(lambda x: masked_attention(x, enc_padding_mask))
        Features_Adder = Lambda(lambda x: sum_and_tanh(x))
        Squeezer_3_2 = Lambda(lambda x: K.squeeze(K.squeeze(x, axis=3), axis=2))
        Expand_Dim_2_2 = Lambda(lambda x: K.expand_dims(K.expand_dims(x, 2), 2))
        Attention_Linear_layer = Linear(attention_vec_size, True)
        # the linear layer used in attention(...),
        # transform decoder_state to decoder_features
        Decoder_Input_to_Cell_Input = Linear(input_size, True)
        Calculate_pgen_Linear_layer = Linear(1, True, activation='sigmoid')
        AttnOutputProjection_Linear_layer = Linear(Cell.output_dim, True)
        Expand_1_1 = Lambda(lambda x: K.expand_dims(K.expand_dims(x, axis=1), axis=1))

        def attention(decoder_state, coverage=None):
            #   Calculate the context vector and attention distribution from the decoder state.
            # Args:
            #   decoder_state: state of the decoder
            #   coverage: Optional. Previous timestep's coverage vector, shape (batch_size, attn_len, 1, 1).
            # Returns:
            #   context_vector: weighted sum of encoder_states
            #   attn_dist: attention distribution
            #   coverage: new coverage vector. shape (batch_size, attn_len, 1, 1)

            decoder_features = Attention_Linear_layer(decoder_state)  # shape (batch_size, attention_vec_size)
            decoder_features = Expand_1_1(decoder_features)  # reshape to (batch_size, 1, 1, attention_vec_size)

            if use_coverage and coverage is not None:
                coverage_features = Coverage_Feature_Extractor(coverage)
                added_features = Features_Adder([encoder_features, decoder_features, coverage_features])
                # added_features: shape (batch_size,attn_length, 1, 1)
                e = Squeezer_3_2(V(added_features))
                # e: shape (batch_size,attn_length)
                # Calculate attention distribution
                attn_dist = Masked_Attention(e)
                # Update coverage vector
                # sum over the input sequence

                coverage = Lambda(lambda x: x[0] + Reshape((-1, 1, 1))(x[1]))([coverage, attn_dist])
            else:
                added_features = Features_Adder([encoder_features, decoder_features])
                # added_features: shape (batch_size,attn_length, 1, 1)
                e = Squeezer_3_2(V(added_features))
                attn_dist = Masked_Attention(e)
                if use_coverage:  # first step of training
                    coverage = Expand_Dim_2_2(attn_dist)  # initialize coverage

            context_vector = Attn_Dist_and_Encoder_States_to_Context_Vector([attn_dist, encoder_states])
            # context_vector = math_ops.reduce_sum(array_ops.reshape(attn_dist,
            #                                                        [batch_size, -1, 1, 1]) * encoder_states,
            #                                                        [1, 2]) # shape (batch_size, attn_size).
            # context_vector = array_ops.reshape(context_vector, [-1, attn_size])

            return context_vector, attn_dist, coverage

        # ####END OF ATTENTION#### #

        # Return values:
        outputs = []
        attn_dists = []
        p_gens = []
        # initial_state is a list/ tuple
        state_h, state_c = initial_state[0], initial_state[1]
        coverage_ret = prev_coverage  # initialize coverage to None or whatever was passed in

        # re-typed to tf.Tensor for backend operations
        context_vector_ret = Lambda(lambda x: K.zeros(shape=(self._batch_size, attn_size)))([])
        # Get a zero-initialized context vector
        if initial_state_attention:
            # Re-calculate the context vector from the previous step
            # so that we can pass it through a linear layer with this step's input
            # to get a modified version of the input
            context_vector_ret, _, coverage_ret = attention(initial_state, coverage_ret)
            # in decode mode, this is what updates the coverage vector
        # otherwise, context_vector & coverage are zero vectors
        for i, inp in enumerate(decoder_inputs):
            transformed_inp = Decoder_Input_to_Cell_Input([inp, context_vector_ret])
            cell_output, state_h, state_c = Cell([transformed_inp, state_h, state_c])
            if i == 0 and initial_state_attention:  # always true in decode mode
                context_vector_ret, attn_dist_ret, _ = attention([state_h, state_c], coverage_ret)
                # don't allow coverage to update
            else:
                context_vector_ret, attn_dist_ret, coverage_ret = attention([state_h, state_c], coverage_ret)
            attn_dists.append(attn_dist_ret)

            if pointer_gen:
                p_gen = Calculate_pgen_Linear_layer([context_vector_ret, state_h, state_c, transformed_inp])
                p_gens.append(p_gen)

            output = AttnOutputProjection_Linear_layer([cell_output, context_vector_ret])
            outputs.append(output)

        print('finished adding attention_decoder for each time step!')
        if coverage_ret is not None:
            coverage_ret = Lambda(lambda x: K.reshape(x, [self._batch_size, -1]))(coverage_ret)

        return outputs, [state_h, state_c], attn_dists, p_gens, coverage_ret

    # ####END OF ATTENTION_DECODER#### #

    def _add_decoder(self, inputs):
        # Args:
        # inputs: inputs to the decoder (word embeddings). (batch_size, emb_dim)
        # stored in list
        _Cell = self.DecoderCell
        _prev_coverage = self.prev_coverage \
            if self.mode == 'decode' and self.coverage else None

        # attention_decoder(inputs, self._dec_in_state, self._enc_states, self._enc_padding_mask, cell,
        # initial_state_attention=(hps.mode=="decode"),
        # pointer_gen=hps.pointer_gen, use_coverage=hps.coverage, prev_coverage=prev_coverage)
        outputs, out_state, attn_dists, p_gens, coverage = \
            self.attention_decoder(decoder_inputs=inputs,
                                   initial_state=self._dec_in_state,
                                   encoder_states=self._enc_states,
                                   enc_padding_mask=self._enc_padding_mask,
                                   Cell=_Cell,
                                   initial_state_attention=(self.mode == 'decode'),
                                   pointer_gen=self.pointer_gen,
                                   use_coverage=self.use_coverage,
                                   prev_coverage=_prev_coverage
                                   )
        return outputs, out_state, attn_dists, p_gens, coverage

    # END OF _ADD_DECODER #

    def _calc_final_dist(self, vocab_dists, attn_dists):
        WeightMultLayer = Lambda(lambda x: x[0] * x[1])
        SupWeightMultLayer = Lambda(lambda x: (1 - x[0]) * x[1])
        DistPlus = Lambda(lambda x: x[0] + x[1])

        vocab_dists = [WeightMultLayer([a, b]) for a, b in zip(self.p_gens, vocab_dists)]
        attn_dists_weighted = [SupWeightMultLayer([a, b]) for a, b in zip(self.p_gens, attn_dists)]

        extra_zeros = Lambda(lambda x: K.zeros(shape=(self._batch_size, self._max_art_oovs), dtype='float32'))([])
        extended_vsize = Lambda(lambda x: self.vsize + x)(self._max_art_oovs)
        vocab_dists_extended = [self.ConcatenateAxis1([dist, extra_zeros]) for dist in vocab_dists]

        # Project the values in the attention distributions onto the appropriate entries in the final distributions
        # This means that if a_i = 0.1 and the ith encoder word is w,
        # and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
        # This is done for each decoder timestep.
        # This is fiddly; we use tf.scatter_nd to do the projection
        shape = [self._batch_size, extended_vsize]

        def preparation(x):
            batch_nums = K.tf.range(0, limit=self._batch_size)  # shape (batch_size)
            batch_nums = K.tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
            attn_len = K.tf.shape(self._enc_batch_extend_vocab)[1]  # number of states we attend over
            batch_nums = K.tf.tile(batch_nums, multiples=[1, attn_len])
            indices = K.tf.stack((batch_nums, self._enc_batch_extend_vocab), axis=2)
            return indices

        indices = Lambda(preparation)([])
        ScatterNdList = [Lambda(
            lambda x: K.tf.scatter_nd(indices, x, shape=shape, name='making_attn_dists_projected_at_step_%d' % _index),
            name='making_attn_dists_projected_at_step_%d' % _index)
            for _index in range(len(attn_dists_weighted))]

        attn_dists_projected = [
            ScatterNdList[_index](copy_dist)
            for _index, copy_dist in enumerate(attn_dists_weighted)]

        final_dists = [DistPlus([a, b]) for a, b in zip(vocab_dists_extended, attn_dists_projected)]

        def _add_epsilon(epsilon=1e-9):
            # return add-epsilon layer
            _AddEpsilon = Lambda(lambda x: x + K.tf.ones_like(x) * epsilon)
            return _AddEpsilon

        AddEpsilon = _add_epsilon()
        final_dists = [AddEpsilon(dist) for dist in final_dists]

        return final_dists, attn_dists

    def setup(self):
        pass

    def build(self):
        # Input: text to be summarized
        # source sequence -> encoder input
        self._enc_batch = Input(name='source', shape=(None,), dtype='int32')
        self._enc_lens = Input(name='source_length', shape=(1,), dtype='int32')
        self._enc_padding_mask = Input(name='encoder_padding_mask', shape=(None,), dtype='float32')

        if self.pointer_gen:
            self._enc_batch_extend_vocab = Input(name='extend_vocab', shape=(None,), dtype='int32')
            # same size within batch
            self._max_art_oovs_inp = Input(name='oovs_in_this_batch', shape=(1,), dtype='int32')

        self._max_art_oovs = Lambda(lambda x: x[0][0])(self._max_art_oovs_inp)  # 1-dim tensor

        # target sequence -> decoder input  [<start>, seq[0], seq[1], ...]
        #                 -> decoder mapping target [seq[0], seq[1], seq[2], ...]
        self._dec_batch = Input(name='decoder_input', shape=(self.config['max_dec_steps'],), dtype='int32')
        self._target_batch = Input(name='target', shape=(self.config['max_dec_steps'],), dtype='int32')
        self._dec_padding_mask = Input(name='decoder_padding_mask',
                                       shape=(self.config['max_dec_steps'],), dtype='float32')
        if self.mode == 'decode':
            self.prev_coverage = Input(name='prev_coverage', shape=(None,))

        self._batch_size = Lambda(lambda x: K.shape(x)[0])(self._enc_batch)

        emb_enc_inputs, emb_dec_inputs = self.get_encoder_decoder_inputs(self._enc_batch, self._dec_batch)
        enc_outputs, forward_h, forward_c, backward_h, backward_c = self.Encoder(emb_enc_inputs)
        encoder_states = [self.ConcatenateLastDim([forward_h, backward_h]),
                          self.ConcatenateLastDim([forward_c, backward_c])]
        # ATTENTION:
        # return_sequence=True
        # enc_outputs.shape: (batchsize, timestep, 2*hiddenunits)
        # encoder_output & encoder_states dimension: 2*hidden_units
        # encoder_states[0] = concatenate (forward.h and backward.h)
        # encoder_states[1] = concatenate (forward.c and backward.c)
        self._enc_states = enc_outputs
        new_state_h, new_state_c = self.ReduceStates(encoder_states[0], encoder_states[1])
        self._dec_in_state = [new_state_h, new_state_c]
        # reduced_output & reduced_states dimension: hidden_units

        decoder_outputs, self._dec_out_state, attn_dists, self.p_gens, self.coverage = \
            self._add_decoder(emb_dec_inputs)

        # decoder_outputs = Lambda(lambda x: K.tf.stack(x, axis=1))(decoder_outputs)  ###

        vocab_scores = [self.DecoderOutputProjector(output_) for output_ in decoder_outputs]
        vocab_dists = [self.SoftmaxforScore(score_) for score_ in vocab_scores]

        if self.pointer_gen:
            final_dists, attn_dists = self._calc_final_dist(vocab_dists=vocab_dists,
                                                            attn_dists=attn_dists)
        else:
            final_dists = vocab_dists

        if self.mode == "decode":
            assert False, 'Decode mode not implemented'
            pass

        stacked_final_dists = Lambda(lambda x: K.tf.stack(x, axis=1), name='stacked_final_dists')(final_dists)
        stacked_attn_dists = Lambda(lambda x: K.tf.stack(x, axis=1), name='stacked_attn_dists')(attn_dists)

        if self.use_coverage:
            self.outputs = [stacked_attn_dists, stacked_final_dists]
        else:
            self.outputs = [stacked_final_dists]

        if self.pointer_gen:
            model = Model(inputs=[self._enc_batch, self._enc_lens,
                                  self._enc_padding_mask, self._enc_batch_extend_vocab,
                                  self._max_art_oovs_inp, self._dec_batch,
                                  self._target_batch, self._dec_padding_mask],
                          outputs=self.outputs)
        else:
            model = Model(inputs=[], outputs=[])

        self.outputs_shapes = [K.shape(x) for x in self.outputs]

        (loss_functions, loss_weights) = ([loss_wrapper(self._dec_padding_mask)[0],
                                           loss_wrapper(self._dec_padding_mask)[1]],
                                          [1., 1.]) if self.use_coverage else (
            [loss_wrapper(self._dec_padding_mask)[1]], [1.])

        model.compile(optimizer=self.adg,
                      loss=loss_functions, loss_weights=loss_weights)

        self.model = model
    # #### END OF BUILD ####


def _mask_and_avg(values, padding_mask):
    """Applies mask to values then returns overall average (a scalar)
    Args:
      values: a list length max_dec_steps containing arrays shape (batch_size).
      padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.
    Returns:
      a scalar
    """
    dec_lens = K.tf.reduce_sum(padding_mask, axis=1, name='dec_lens')  # shape batch_size. float32
    values_per_step = []
    for dec_step, v in enumerate(values):
        values_per_step.append(v * padding_mask[:, dec_step])
    values_per_ex = sum(values_per_step) / dec_lens  # shape (batch_size); normalized value for each batch member
    return K.tf.reduce_mean(values_per_ex, name='reduce_mean_in_mask_avg')  # overall average


def loss_wrapper(mask):
    def calc_loss_at_timestep_t(range_batch, t, dist_at_t, _target_batch):
        # Return:
        # losses: loss of all samples in a batch at time step t
        targets = K.tf.strided_slice(_target_batch, [0, t], [K.tf.shape(_target_batch)[0], t + 1], shrink_axis_mask=2,
                                     name='slicing_for_targets_in_calc_loss_at_timestep_t')  # shape: (batch_size, )
        indices = K.tf.stack((range_batch, targets), axis=1)  # shape (batch_size, 2)
        gold_probs = K.tf.gather_nd(dist_at_t, indices)  # shape (batch_size). prob of correct words on this step
        losses = -K.tf.log(gold_probs)
        return losses

    def _loss(y_true, y_pred):
        # Params:
        # y_pred : final_dists, distributions of words, shape (batch_size, time_steps, vocab_size) (float)
        # y_true : indices of true words, shape (batch_size, time_steps, ) (int)

        y_true = K.tf.cast(y_true[:, :, 0], 'int32', 'cast_to_int_in_loss')

        loss_per_step = []
        _batchsize = K.shape(y_pred)[0]
        batch_nums = K.tf.range(0, limit=_batchsize)  # shape: (batch_size, )
        for dec_step, dist in enumerate(K.tf.unstack(y_pred, axis=1)):
            losses = calc_loss_at_timestep_t(batch_nums, dec_step, dist, y_true)
            loss_per_step.append(losses)
        _loss_ret = _mask_and_avg(loss_per_step, padding_mask=mask)

        return _loss_ret

    def _coverage_loss(y_true, y_pred):
        # Params:
        # y_pred : attn_dists, distributions of words, shape (batch_size, time_steps, vocab_size) (float)
        # y_true : indices of true words, shape (batch_size, time_steps, vocab_size ) (int)
        # keras requires y_true and y_pred to be the same shape,
        # thus y_true is repeated vocab_size times on the last dim

        _y_pred = K.tf.unstack(y_pred, axis=1, name='unstacking_attn_dists_in_coverage_loss')
        coverage = K.tf.zeros_like(_y_pred[0])
        covlosses = []
        for a in _y_pred:
            covloss = K.tf.reduce_sum(K.tf.minimum(a, coverage), [1])  # calculate the coverage loss for this step
            covlosses.append(covloss)
            coverage += a  # update the coverage vector
        _coverage_loss_ret = _mask_and_avg(covlosses, padding_mask=mask)
        return _coverage_loss_ret

    return _coverage_loss, _loss


#  Utility
def sum_and_tanh(X):
    # Input: X: list of tensors
    # Return: tanh(X[0] + X[1] + ...)
    tmp = X[0]
    for x in X[1:]:
        tmp += x
    tmp = activations.tanh(tmp)
    return tmp


def attn_dist_and_encoder_states_to_context_vector(X, attn_size):
    # X[0] : attn_dist; X[1] : encoder_states
    reshaped = K.reshape(X[0], (K.shape(X[0])[0],) + (-1, 1, 1)) * X[1]
    con_vec = K.sum(reshaped, [1, 2])
    con_vec = K.reshape(con_vec, (K.shape(X[0])[0], attn_size))
    return con_vec


def masked_attention(e, enc_padding_mask):
    # TODO:
    # epsilon ???
    """Take softmax of e then apply enc_padding_mask and re-normalize"""
    attn_dist = K.softmax(e)
    attn_dist *= enc_padding_mask
    masked_sums = K.sum(attn_dist, axis=1)
    return attn_dist / K.reshape(masked_sums, [-1, 1])


def main():
    ConfigtoFeed = {'mode': 'train',
                    'use_coverage': True,
                    'pointer_gen': True,
                    'train_embed': True,
                    'embed_size': 300,
                    'vocab_size': 100000,
                    'rand_unif_init_mag': 0.01,
                    'trunc_norm_init_std': 0.01,
                    'hidden_units': 128,
                    'max_dec_steps': 100,
                    'cov_loss_wt': 1.0,
                    'embed': None
                    }
    hps, hps_dict = get_hps()
    configs = {}
    for kys in ConfigtoFeed:
        if kys in hps_dict:
            configs[kys] = hps_dict[kys]
        else:
            configs[kys] = ConfigtoFeed[kys]

    MD = SummarizationModel(None, configs, hps)
    MD.build()
    vocab_path = '/path/to/vocab'
    train_data_path = '/path/to/train*'
    valid_data_path = '/path/to/val*'
    single_pass = True
    vocab = Vocab(vocab_path, hps.vocab_size)

    train_batcher = Batcher(train_data_path, vocab, hps, single_pass=single_pass)
    TRAIN_GEN = SummarizationGenerator(hps, train_batcher)
    train_generator = TRAIN_GEN.get_batch_generator()

    valid_batcher = Batcher(valid_data_path, vocab, hps, single_pass=False)
    VALID_GEN = SummarizationGenerator(hps, valid_batcher)
    valid_generator = VALID_GEN.get_batch_generator()

    print("TRAINING IN PROGRESS...")

    for i_e in range(100):

        history = MD.model.fit_generator(train_generator,
                                         steps_per_epoch=200,
                                         epochs=1,
                                         shuffle=False,
                                         verbose=0)  # ~3200 samples per call
        for x in history.history:
            print('training:', history.history[x])

        validation = MD.model.evaluate_generator(valid_generator, steps=100, verbose=0)  # ~1600 samples per call
        print('validation:', validation)


# Where to find data
tf.app.flags.DEFINE_string('data_path', '/mnt/E/WORK/DATA/CNN_DM/finished_files/chunked/train*',
                           'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('vocab_path', '/mnt/E/WORK/DATA/CNN_DM/finished_files/vocab',
                           'Path expression to text vocabulary file.')

# Important settings
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of train/eval/decode')
tf.app.flags.DEFINE_boolean('single_pass', False,
                            'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')

# Where to save output
tf.app.flags.DEFINE_string('log_root', '', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name', '',
                           'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('batch_size', 16, 'minibatch size')  # originally 16
tf.app.flags.DEFINE_integer('max_enc_steps', 400, 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_dec_steps', 100, 'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('min_dec_steps', 35,
                            'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.app.flags.DEFINE_integer('vocab_size', 50000,
                            'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_float('lr', 0.15, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')
# Pointer-generator or baseline model
tf.app.flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator model. If False, use baseline model.')
# Coverage hyperparameters
tf.app.flags.DEFINE_boolean('use_coverage', False,
                            'Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
tf.app.flags.DEFINE_float('cov_loss_wt', 1.0,
                          'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')
# Utility flags, for restoring and changing checkpoints
tf.app.flags.DEFINE_boolean('convert_to_coverage_model', False,
                            'Convert a non-coverage model to a coverage model. Turn this on and run in train mode. Your current training model will be copied to a new version (same name with _cov_init appended) that will be ready to run with coverage flag turned on, for the coverage training stage.')
tf.app.flags.DEFINE_boolean('restore_best_model', False,
                            'Restore the best model in the eval/ dir and save it in the train/ dir, ready to be used for further training. Useful for early stopping, or if your training checkpoint has become corrupted with e.g. NaN values.')
# Debugging. See https://www.tensorflow.org/programmers_guide/debugger
tf.app.flags.DEFINE_boolean('debug', False, "Run in tensorflow's debug mode (watches for NaN/inf values)")


def get_hps():
    hparam_list = ['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm',
                   'vocab_size',
                   'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps', 'use_coverage',
                   'cov_loss_wt',
                   'pointer_gen']
    hps_dict = {}
    for key, val in tf.app.flags.FLAGS.__flags.items():  # for each flag
        if key in hparam_list:  # if it's in the list
            hps_dict[key] = val.value  # add it to the dict
    hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)
    return hps, hps_dict


if __name__ == '__main__':
    main()
    pass
