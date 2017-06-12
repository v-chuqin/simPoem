#coding=utf-8
'''
Build a neural machine translation model with soft attention
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import ipdb
import numpy
import copy

import os
import warnings
import sys
import time

from collections import OrderedDict

from data_iterator import TextKeywordIterator, TestTextKeywordIterator
import pudb


profile = False


# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]


# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=0.5, n=1,
                                     dtype=state_before.dtype),
        state_before * 0.5)
    return proj


# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]

    return params

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'gru_cond': ('param_init_gru_cond', 'gru_cond_layer'),
          }


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


# some utilities
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


def tanh(x):
    return tensor.tanh(x)


def linear(x):
    return x


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out

def prepare_test_source(previous_source_seq, dictionary, line, pred_target):
    if 0 == line:
        x = numpy.zeros((previous_source_seq.shape[0] + 1, 1)).astype('int64')
        x[:previous_source_seq.shape[0], :] =  previous_source_seq
        x[-1, 0] = 0
    elif 1 == line:
        x = numpy.zeros((pred_target.shape[0], 1)).astype('int64')
        x[:pred_target.shape[0] -1, :] = pred_target[:-1, :]
        x[-1, 0] = 0
    else:
        x = numpy.zeros((previous_source_seq.shape[0] + pred_target.shape[0] - 2, 1)).astype('int64')
        count = 0
        for i in range(previous_source_seq.shape[0] - 2):
            x[count, 0] = previous_source_seq[i, 0]
            count = count + 1
        if 2 == line:
            sep_num = dictionary['，']
        else:
            sep_num = dictionary['。']

        x[count, 0] = sep_num
        count = count + 1
        for i in range(1, pred_target.shape[0] - 1):
            x[count, 0] = pred_target[i, 0]
            count = count + 1
        x[-1, 0] = 0
    return x

# batch preparation
def prepare_data(seqs_x, seqs_y, seqs_k, maxlen=None, n_words_src=30000,
                 n_words=30000, phase = 'train'):
    # x: a list of sentences

    if 'train' == phase:
        lengths_k = [len(k) for k in seqs_k]
        lengths_x = [len(s) for s in seqs_x]
        lengths_y = [len(s) for s in seqs_y]
        if maxlen is not None:
            new_seqs_x = []
            new_seqs_y = []
            new_seqs_k = []
            new_lengths_x = []
            new_lengths_y = []
            new_lengths_k = []
            for l_x, s_x, l_y, s_y, l_k, s_k in zip(lengths_x, seqs_x, 
                                                    lengths_y, seqs_y,
                                                    lengths_k, seqs_k):
                if l_x < maxlen and l_y < maxlen and l_k < maxlen:
                    new_seqs_x.append(s_x)
                    new_lengths_x.append(l_x)
                    new_seqs_y.append(s_y)
                    new_lengths_y.append(l_y)
                    new_seqs_k.append(s_k)
                    new_lengths_k.append(l_k)
            lengths_x = new_lengths_x
            seqs_x = new_seqs_x
            lengths_y = new_lengths_y
            seqs_y = new_seqs_y
            lengths_k = new_lengths_k
            seqs_k = new_seqs_k

            if len(lengths_x) < 1 or len(lengths_y) < 1 or len(lengths_k) < 1:
                return None, None, None, None, None, None

        n_samples = len(seqs_x)
        maxlen_x = numpy.max(lengths_x) + 1
        maxlen_y = numpy.max(lengths_y) + 1
        maxlen_k = numpy.max(lengths_k)

        x = numpy.zeros((maxlen_x, n_samples)).astype('int64')
        y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
        k = numpy.zeros((maxlen_k, n_samples)).astype('int64')
        x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
        y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
        k_mask = numpy.zeros((maxlen_k, n_samples)).astype('float32')
        for idx, [s_x, s_y, s_k] in enumerate(zip(seqs_x, seqs_y, seqs_k)):
            x[:lengths_x[idx], idx] = s_x
            x_mask[:lengths_x[idx]+1, idx] = 1.
            y[:lengths_y[idx], idx] = s_y
            y_mask[:lengths_y[idx]+1, idx] = 1.
            k[:lengths_k[idx], idx] = s_k
            k_mask[:lengths_k[idx], idx] = 1.
        return x, x_mask, y, y_mask, k, k_mask
    else:
        lengths_k_lines = [len(k) for k in seqs_k]
        n_samples = len(seqs_k)
        max_lengths_k_lines = numpy.max(lengths_k_lines)
        lengths_k_w = []
        max_lengths_k_w = 0
        for i in range(len(seqs_k)):
            lengths_k_w.append([len(line) for line in seqs_k[i]])
            max_line_length_w = numpy.max(lengths_k_w[i])
            max_lengths_k_w = max_line_length_w > max_lengths_k_w \
                    and max_line_length_w or max_lengths_k_w
        k = numpy.zeros((max_lengths_k_w, n_samples, max_lengths_k_lines)).astype('int64')
        k_mask = numpy.zeros((max_lengths_k_w, n_samples, max_lengths_k_lines)).astype('float32')
        for i in range(len(seqs_k)):
            for j in range(len(seqs_k[i])):
                k[:lengths_k_w[i][j], i, j] = seqs_k[i][j]
                k_mask[:lengths_k_w[i][j], i, j] = 1.
        return k, k_mask
                



# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None,
                       ortho=True):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams, state_below, options, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(
        tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
        tparams[_p(prefix, 'b')])


# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']

    # embedding to gates transformation weights, biases
    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')

    # recurrent transformation weights for gates
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U

    # embedding to hidden state proposal weights, biases
    Wx = norm_weight(nin, dim)
    params[_p(prefix, 'Wx')] = Wx
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux = ortho_weight(dim)
    params[_p(prefix, 'Ux')] = Ux

    return params


def gru_layer(tparams, state_below, options, prefix='gru', mask=None, init_states=None,
              **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # state_below is the input word embeddings
    # input to the gates, concatenated
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
        tparams[_p(prefix, 'b')]
    # input to compute the hidden state proposal
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + \
        tparams[_p(prefix, 'bx')]

    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(m_, x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        # compute the hidden state proposal
        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        # hidden state proposal
        h = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx]
    if init_states is None:
        init_states = [tensor.alloc(0., n_samples, dim)]
    _step = _step_slice
    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')]]

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_states,
                                non_sequences=shared_vars,
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile,
                                strict=True)
    rval = [rval]
    return rval


# Conditional GRU layer with Attention
def param_init_gru_cond(options, params, prefix='gru_cond',
                        nin=None, dim=None, dimctx=None,
                        nin_nonlin=None, dim_nonlin=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']
    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim

    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim_nonlin),
                           ortho_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'U')] = U

    Wx = norm_weight(nin_nonlin, dim_nonlin)
    params[_p(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim_nonlin)
    params[_p(prefix, 'Ux')] = Ux
    params[_p(prefix, 'bx')] = numpy.zeros((dim_nonlin,)).astype('float32')

    U_nl = numpy.concatenate([ortho_weight(dim_nonlin),
                              ortho_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'U_nl')] = U_nl
    params[_p(prefix, 'b_nl')] = numpy.zeros((2 * dim_nonlin,)).astype('float32')

    Ux_nl = ortho_weight(dim_nonlin)
    params[_p(prefix, 'Ux_nl')] = Ux_nl
    params[_p(prefix, 'bx_nl')] = numpy.zeros((dim_nonlin,)).astype('float32')

    # context to LSTM
    Wc = norm_weight(dimctx, dim*2)
    params[_p(prefix, 'Wc')] = Wc

    Wcx = norm_weight(dimctx, dim)
    params[_p(prefix, 'Wcx')] = Wcx

    # attention: combined -> hidden
    W_comb_att = norm_weight(dim, dimctx)
    params[_p(prefix, 'W_comb_att')] = W_comb_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[_p(prefix, 'Wc_att')] = Wc_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1)
    params[_p(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_tt')] = c_att

    return params


def gru_cond_layer(tparams, state_below, options, prefix='gru',
                   mask=None, context=None, one_step=False,
                   init_memory=None, init_state=None,
                   context_mask=None,
                   **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Wcx')].shape[1]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = tensor.dot(context, tparams[_p(prefix, 'Wc_att')]) +\
        tparams[_p(prefix, 'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) +\
        tparams[_p(prefix, 'bx')]
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) +\
        tparams[_p(prefix, 'b')]

    def _step_slice(m_, x_, xx_, h_, ctx_, alpha_, pctx_, cc_,
                    U, Wc, W_comb_att, U_att, c_tt, Ux, Wcx,
                    U_nl, Ux_nl, b_nl, bx_nl):
        preact1 = tensor.dot(h_, U)
        preact1 += x_
        preact1 = tensor.nnet.sigmoid(preact1)

        r1 = _slice(preact1, 0, dim)
        u1 = _slice(preact1, 1, dim)

        preactx1 = tensor.dot(h_, Ux)
        preactx1 *= r1
        preactx1 += xx_

        h1 = tensor.tanh(preactx1)

        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_

        # attention
        pstate_ = tensor.dot(h1, W_comb_att)
        pctx__ = pctx_ + pstate_[None, :, :]
        #pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__, U_att)+c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context

        preact2 = tensor.dot(h1, U_nl)+b_nl
        preact2 += tensor.dot(ctx_, Wc)
        preact2 = tensor.nnet.sigmoid(preact2)

        r2 = _slice(preact2, 0, dim)
        u2 = _slice(preact2, 1, dim)

        preactx2 = tensor.dot(h1, Ux_nl)+bx_nl
        preactx2 *= r2
        preactx2 += tensor.dot(ctx_, Wcx)

        h2 = tensor.tanh(preactx2)

        h2 = u2 * h1 + (1. - u2) * h2
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1

        return h2, ctx_, alpha.T  # pstate_, preact, preactx, r, u

    seqs = [mask, state_below_, state_belowx]
    #seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Wc')],
                   tparams[_p(prefix, 'W_comb_att')],
                   tparams[_p(prefix, 'U_att')],
                   tparams[_p(prefix, 'c_tt')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p(prefix, 'Wcx')],
                   tparams[_p(prefix, 'U_nl')],
                   tparams[_p(prefix, 'Ux_nl')],
                   tparams[_p(prefix, 'b_nl')],
                   tparams[_p(prefix, 'bx_nl')]]

    if one_step:
        rval = _step(*(seqs + [init_state, None, None, pctx_, context] +
                       shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[2]),
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[0])],
                                    non_sequences=[pctx_, context]+shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval


# initialize all parameters
def init_params(options):
    params = OrderedDict()

    # embedding
    params['Wemb'] = norm_weight(options['n_words_src'], options['dim_word'])
    params['Wemb_dec'] = norm_weight(options['n_words'], options['dim_word'])

    # encoder: bidirectional RNN
    params = get_layer(options['encoder'])[0](options, params,
                                              prefix='encoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'])
    params = get_layer(options['encoder'])[0](options, params,
                                              prefix='encoder_r',
                                              nin=options['dim_word'],
                                              dim=options['dim'])
    ctxdim = 2 * options['dim']

    # init_state, init_cell
    params = get_layer('ff')[0](options, params, prefix='ff_state',
                                nin=ctxdim, nout=options['dim'])
    # decoder
    params = get_layer(options['decoder'])[0](options, params,
                                              prefix='decoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'],
                                              dimctx=ctxdim)
    # readout
    params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm',
                                nin=options['dim'], nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_prev',
                                nin=options['dim_word'],
                                nout=options['dim_word'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_ctx',
                                nin=ctxdim, nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit',
                                nin=options['dim_word'],
                                nout=options['n_words'])

    return params


# build a training model
def build_model(tparams, options):
    opt_ret = dict()

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')
    y = tensor.matrix('y', dtype='int64')
    y_mask = tensor.matrix('y_mask', dtype='float32')
    keyword = tensor.matrix('keyword', dtype = 'int64')
    keyword_mask = tensor.matrix('keyowrd_mask', dtype = 'float32')

    keywordr = keyword[::-1]
    keywordr_mask = keyword_mask[::-1]
    n_timesteps = keyword.shape[0]
    n_samples = keyword.shape[1]
    keyword_emb = tparams['Wemb'][keyword.flatten()]
    keyword_emb = keyword_emb.reshape([n_timesteps, n_samples, options['dim_word']])
    proj = get_layer(options['encoder'])[1](tparams, keyword_emb, options,
                                            prefix='encoder',
                                            mask=keyword_mask)
    # word embedding for backward rnn (source)
    keyword_embr = tparams['Wemb'][keywordr.flatten()]
    keyword_embr = keyword_embr.reshape([n_timesteps, n_samples, options['dim_word']])
    projr = get_layer(options['encoder'])[1](tparams, keyword_embr, options,
                                             prefix='encoder_r',
                                             mask=keywordr_mask)
    keyword_hidden = concatenate([proj[0][-1], projr[0][-1]], axis = proj[0].ndim - 2)
    # for the backward rnn, we just need to invert x and x_mask
    xr = x[::-1]
    xr_mask = x_mask[::-1]

    n_timesteps = x.shape[0]
    n_timesteps_trg = y.shape[0]
    n_samples = x.shape[1]

    # word embedding for forward rnn (source)
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder',
                                            mask=x_mask)
    # word embedding for backward rnn (source)
    embr = tparams['Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word']])
    projr = get_layer(options['encoder'])[1](tparams, embr, options,
                                             prefix='encoder_r',
                                             mask=xr_mask)

    # context will be the concatenation of forward and backward rnns
    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)
    temp = tensor.alloc(1., 1, n_samples)
    ctx = concatenate([ctx, keyword_hidden[None, :, :]], axis = 0)
    ex_x_mask = concatenate([temp, x_mask], axis = 0)
    # mean of the context (across time) will be used to initialize decoder rnn
    ctx_mean = (ctx * ex_x_mask[:, :, None]).sum(0) / ex_x_mask.sum(0)[:, None]

    # or you can use the last state of forward + backward encoder rnns
    # ctx_mean = concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)

    # initial decoder state
    init_state = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    # word embedding (target), we will shift the target sequence one time step
    # to the right. This is done because of the bi-gram connections in the
    # readout and decoder rnn. The first target will be all zeros and we will
    # not condition on the last output.
    emb = tparams['Wemb_dec'][y.flatten()]
    emb = emb.reshape([n_timesteps_trg, n_samples, options['dim_word']])
    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted

    # decoder - pass through the decoder conditional gru with attention
    proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                            prefix='decoder',
                                            mask=y_mask, context=ctx,
                                            context_mask=ex_x_mask,
                                            one_step=False,
                                            init_state=init_state)
    # hidden states of the decoder gru
    proj_h = proj[0]

    # weighted averages of context, generated by attention module
    ctxs = proj[1]

    # weights (alignment matrix)
    opt_ret['dec_alphas'] = proj[2]

    # compute word probabilities
    logit_lstm = get_layer('ff')[1](tparams, proj_h, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)
    if options['use_dropout']:
        logit = dropout_layer(logit, use_noise, trng)
    logit = get_layer('ff')[1](tparams, logit, options,
                               prefix='ff_logit', activ='linear')
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1],
                                               logit_shp[2]]))

    # cost
    y_flat = y.flatten()
    y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words'] + y_flat
    cost = -tensor.log(probs.flatten()[y_flat_idx])
    cost = cost.reshape([y.shape[0], y.shape[1]])
    cost = (cost * y_mask).sum(0)

    return trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost, keyword, keyword_mask, keyword_hidden


# build a sampler
def build_sampler(tparams, options, trng, use_noise):
    keyword = tensor.matrix('x', dtype='int64')
    keyword_mask = tensor.matrix('keyowrd_mask', dtype = 'float32')
    
    keywordr = keyword[::-1]
    keywordr_mask = keyword_mask[::-1]
    n_timesteps = keyword.shape[0]
    n_samples = keyword.shape[1]

    # word embedding (source), forward and backward
    keyword_emb = tparams['Wemb'][keyword.flatten()]
    keyword_emb = keyword_emb.reshape([n_timesteps, 
                                       n_samples, 
                                       options['dim_word']])
    keyword_embr = tparams['Wemb'][keywordr.flatten()]
    keyword_embr = keyword_embr.reshape([n_timesteps, 
                                         n_samples, 
                                         options['dim_word']])

    # encoder
    proj = get_layer(options['encoder'])[1](tparams, keyword_emb, options,
                                            prefix='encoder', 
                                            mask = keyword_mask)
    projr = get_layer(options['encoder'])[1](tparams, keyword_embr, options,
                                             prefix='encoder_r', 
                                             mask = keywordr_mask)
    keyword_hidden = concatenate([proj[0][-1], projr[0][-1]], axis = proj[0].ndim - 2)

    x = tensor.matrix('x', dtype='int64')
    xr = x[::-1]
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # word embedding (source), forward and backward
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    embr = tparams['Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word']])

    # encoder
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder')
    projr = get_layer(options['encoder'])[1](tparams, embr, options,
                                             prefix='encoder_r')

    # concatenate forward and backward rnn hidden states
    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)
    ctx = concatenate([ctx, keyword_hidden[None, :, :]], axis = 0)

    # get the input for decoder rnn initializer mlp
    ctx_mean = ctx.mean(0)
    # ctx_mean = concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)
    init_state = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    print 'Building f_init...',
    outs = [init_state, ctx]
    f_init = theano.function([x, keyword, keyword_mask], 
                             outs, 
                             name='f_init', 
                             profile=profile)
    print 'Done'

    # x: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')

    # if it's the first word, emb should be all zero and it is indicated by -1
    emb = tensor.switch(y[:, None] < 0,
                        tensor.alloc(0., 1, tparams['Wemb_dec'].shape[1]),
                        tparams['Wemb_dec'][y])

    # apply one step of conditional gru with attention
    proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                            prefix='decoder',
                                            mask=None, context=ctx,
                                            one_step=True,
                                            init_state=init_state)
    # get the next hidden state
    next_state = proj[0]

    # get the weighted averages of context for this target word y
    ctxs = proj[1]

    logit_lstm = get_layer('ff')[1](tparams, next_state, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)
    if options['use_dropout']:
        logit = dropout_layer(logit, use_noise, trng)
    logit = get_layer('ff')[1](tparams, logit, options,
                               prefix='ff_logit', activ='linear')

    # compute the softmax probability
    next_probs = tensor.nnet.softmax(logit)

    # sample from softmax distribution to get the sample
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # compile a function to do the whole thing above, next word probability,
    # sampled word for the next target, next hidden state to be used
    print 'Building f_next..',
    inps = [y, ctx, init_state]
    outs = [next_probs, next_sample, next_state]
    f_next = theano.function(inps, outs, name='f_next', profile=profile)
    print 'Done'

    return f_init, f_next


# generate sample, either with stochastic sampling or beam search. Note that,
# this function iteratively calls f_init and f_next functions.
def gen_sample(tparams, f_init, f_next, x, keyword, keyword_mask,
               options, trng=None, k=1, maxlen=30,
               stochastic=True, argmax=False):

    # k is the beam size we have
    if k > 1:
        assert not stochastic, \
            'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []

    # get initial state of decoder rnn and encoder context
    ret = f_init(x, keyword, keyword_mask)
    next_state, ctx0 = ret[0], ret[1]
    next_w = -1 * numpy.ones((1,)).astype('int64')  # bos indicator

    for ii in xrange(maxlen):
        ctx = numpy.tile(ctx0, [live_k, 1])
        inps = [next_w, ctx, next_state]
        ret = f_next(*inps)
        next_p, next_w, next_state = ret[0], ret[1], ret[2]

        if stochastic:
            if argmax:
                nw = next_p[0].argmax()
                next_w = numpy.array([nw])
            else:
                nw = next_w[0]
            sample.append(nw)
            sample_score -= numpy.log(next_p[0, nw])
            if nw == 0:
                break
        else:
            cand_scores = hyp_scores[:, None] - numpy.log(next_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k-dead_k)]

            voc_size = next_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_hyp_states = []

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append(copy.copy(next_state[ti]))

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = numpy.array(hyp_states)

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

    return sample, sample_score


# calculate the log probablities on a given corpus using translation model
def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=True):
    probs = []

    n_done = 0

    for x, y, keyword in iterator:
        n_done += len(x)

        x, x_mask, y, y_mask, \
                keyword, keyword_mask = prepare_data(x, y, keyword,
                                        n_words_src=options['n_words_src'],
                                        n_words=options['n_words'])

        pprobs = f_log_probs(x, x_mask, y, y_mask, keyword, keyword_mask)
        for pp in pprobs:
            probs.append(pp)

        if numpy.isnan(numpy.mean(probs)):
            ipdb.set_trace()

        if verbose:
            print >>sys.stderr, '%d samples computed' % (n_done)

    return numpy.array(probs)


# optimizers
# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adam(lr, tparams, grads, inp, cost, beta1=0.9, beta2=0.999, e=1e-8):

    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup, profile=profile)

    updates = []

    t_prev = theano.shared(numpy.float32(0.))
    t = t_prev + 1.
    lr_t = lr * tensor.sqrt(1. - beta2**t) / (1. - beta1**t)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0., p.name + '_mean')
        v = theano.shared(p.get_value() * 0., p.name + '_variance')
        m_t = beta1 * m + (1. - beta1) * g
        v_t = beta2 * v + (1. - beta2) * g**2
        step = lr_t * m_t / (tensor.sqrt(v_t) + e)
        p_t = p - step
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((t_prev, t))

    f_update = theano.function([lr], [], updates=updates,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rg2up,
                                    profile=profile)

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads, running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rgup+rg2up,
                                    profile=profile)

    updir = [theano.shared(p.get_value() * numpy.float32(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(itemlist(tparams), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def sgd(lr, tparams, grads, x, mask, y, cost):
    gshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    profile=profile)

    pup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]
    f_update = theano.function([lr], [], updates=pup, profile=profile)

    return f_grad_shared, f_update


def run(options):
    dictionaries = [
            '%s/data/%s' % (options['basedir'], 
                            options['source_dict']), 
            '%s/data/%s' % (options['basedir'], 
                            options['target_dict'])]
    reference_dictionary = '%s/data/%s' % (options['basedir'],
                                           options['reference_dictionary'])
    saveto_dir='%s/models/%s/' % (options['basedir'], 
                                  options['saveto'].split('.')[0])
    saveto = '%s/%s' % (saveto_dir, options['saveto'])
    reload_ = options['reload_flag']
    reload_iter = options['reload_iter']
    mode = options['mode']
    datasets = [
            '%s/data/%s' % (options['basedir'], 
                            options['train_source']), 
            '%s/data/%s' % (options['basedir'], 
                            options['train_target'])]
    valid_datasets = [
            '%s/data/%s' % (options['basedir'], 
                            options['val_source']), 
            '%s/data/%s' % (options['basedir'], 
                            options['val_target'])]
    test_datasets = [
            '%s/data/%s' % (options['basedir'], 
                            options['test_source']), 
            '%s/results/%s' % (options['basedir'], 
                            options['test_target'])]
    reference = [
            '%s/data/%s' % (options['basedir'], 
                            options['train_reference']), 
            '%s/data/%s' % (options['basedir'], 
                            options['val_reference'])]
    keywords = [
            '%s/data/%s' % (options['basedir'], 
                            options['train_keyword']), 
            '%s/data/%s' % (options['basedir'], 
                            options['val_keyword'])]
    image_dirs = [
            '%s/data/%s' % (options['basedir'], 
                            options['conv5_dir']), 
            '%s/data/%s' % (options['basedir'], 
                            options['fc7_dir'])]
    n_words_src = options['n_words_src']
    n_words = options['n_words']
    n_words_reference = options['n_words_reference']
    batch_size = options['batch_size']
    maxlen = options['maxlen']
    valid_batch_size = options['valid_batch_size']
    decay_c = options['decay_c']
    alpha_c = options['alpha_c']
    clip_c = options['clip_c']
    optimizer = options['optimizer']
    validFreq = options['validFreq']
    saveFreq = options['saveFreq']
    sampleFreq = options['sampleFreq']
    dispFreq = options['dispFreq']
    max_epochs = options['max_epochs']
    lrate = options['lrate']
    overwrite = options['overwrite']
    stochastic = options['stochastic']
    argmax = options['argmax']
    validSize = options['validSize']
    patience = options['patience']
    finish_after = options['finish_after']   
    train_skip = options['train_skip']
    valid_skip = options['valid_skip']

    # load dictionaries and invert them
    worddicts = [None] * len(dictionaries)
    worddicts_r = [None] * len(dictionaries)
    for ii, dd in enumerate(dictionaries):
        with open(dd, 'rb') as f:
            worddicts[ii] = pkl.load(f)
        worddicts_r[ii] = dict()
        for kk, vv in worddicts[ii].iteritems():
            worddicts_r[ii][vv] = kk

    # reload options
    if reload_:
        option_file = '%s.pkl' % saveto
        if reload_iter >= 0:
            option_file = '{}.iter{}.npz.pkl'.format(
                    os.path.splitext(saveto)[0], reload_iter)
        if os.path.exists(option_file):
            print 'Reloading model options from ', option_file
            with open(option_file, 'rb') as f:
                options = pkl.load(f)

    print 'Loading data'
    if 'train' == mode:
        train = TextKeywordIterator(datasets[0], datasets[1], 
                             keywords[0],
                             dictionaries[0], dictionaries[1],
                             n_words_source=n_words_src, n_words_target=n_words,
                             batch_size=batch_size,
                             maxlen=maxlen)
        valid = TextKeywordIterator(valid_datasets[0], valid_datasets[1],
                             keywords[1],
                             dictionaries[0], dictionaries[1],
                             n_words_source=n_words_src, n_words_target=n_words,
                             batch_size=valid_batch_size,
                             maxlen=maxlen)
    else:
        test = TestTextKeywordIterator(test_datasets[0],
                             dictionaries[0],
                             n_words_source=n_words_src,
                             batch_size=1,
                             maxlen=maxlen)

    print 'Building model'
    params = init_params(options)
    # reload parameters
    if reload_: 
        load_file = saveto
        if reload_iter >= 0:
            load_file = '{}.iter{}.npz'.format(
                os.path.splitext(saveto)[0], reload_iter)

        if os.path.exists(load_file):
            print 'Reloading model parameters from ', load_file
            params = load_params(load_file, params)

    tparams = init_tparams(params)

    trng, use_noise, \
        x, x_mask, y, y_mask, \
        opt_ret, \
        cost, keyword, keyword_mask, \
        keyword_hidden \
        = build_model(tparams, options)
    inps = [x, x_mask, y, y_mask, keyword, keyword_mask]

    print 'Building sampler'
    f_init, f_next = build_sampler(tparams, options, trng, use_noise)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile)
    print 'Done'

    cost = cost.mean()

    # apply L2 regularization on weights
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # regularize the alpha weights
    if alpha_c > 0. and not options['decoder'].endswith('simple'):
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * (
            (tensor.cast(y_mask.sum(0)//x_mask.sum(0), 'float32')[:, None] -
             opt_ret['dec_alphas'].sum(0))**2).sum(1).mean()
        cost += alpha_reg

    # after all regularizers - compile the computational graph for cost
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=profile)
    print 'Done'

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print 'Done'

    # apply gradient clipping here
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        grads = new_grads

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)
    print 'Done'

    print 'Optimization'

    best_p = None
    bad_counter = 0
    uidx = 0
    estop = False
    history_errs = []
    # reload history
    if reload_: 
        load_file = saveto
        if reload_iter >= 0:
            load_file = '{}.iter{}.npz'.format(
                os.path.splitext(saveto)[0], reload_iter)
        if os.path.exists(load_file):
            rmodel = numpy.load(load_file)
            history_errs = list(rmodel['history_errs'])
            if 'uidx' in rmodel:
                uidx = rmodel['uidx']

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size
    if sampleFreq == -1:
        sampleFreq = len(train[0])/batch_size

    # pudb.set_trace()
    if 'train' == mode:
        for eidx in xrange(max_epochs):
            n_samples = 0

            prev_disp = time.time()
            for x, y, keyword in train:
                n_samples += len(x)
                # use_noise.set_value(1.)

                x, x_mask, y, y_mask, \
                keyword, keyword_mask = prepare_data(x, y, keyword, 
                                                     maxlen=maxlen,
                                                     n_words_src=n_words_src,
                                                     n_words=n_words)
                if x is None:
                    print 'Minibatch with zero sample under length ', maxlen
                    uidx -= 1
                    continue

                ud_start = time.time()

                # compute cost, grads and copy grads to shared variables
                cost = f_grad_shared(x, x_mask, 
                                     y, y_mask, 
                                     keyword, keyword_mask)

                # check for bad numbers, 
                # usually we remove non-finite elements
                # and continue training - but not done here
                if numpy.isnan(cost) or numpy.isinf(cost):
                    print 'NaN detected'
                    saveto_nan = '{}.iter{}_nan.npz'.format(
                        os.path.splitext(saveto)[0], uidx)
                    pkl.dump(options, open('%s.pkl' % saveto_nan, 'wb'))
                    numpy.savez(saveto_nan, history_errs=history_errs,
                                uidx=uidx, **unzip(tparams))
                    print('n_samples: ', n_samples)
                    ipdb.set_trace()

                    return 1., 1., 1.
                # do the update on parameters
                f_update(lrate)

                ud = time.time() - ud_start


                # verbose
                if numpy.mod(uidx, dispFreq) == 0:
                    duration_disp = time.time() - prev_disp
                    prev_disp = time.time()
                    print 'Epoch ', eidx, \
                          'Update ', uidx, \
                          'Cost ', cost, \
                          'UD ', ud, \
                          'Duration_disp ', duration_disp

                # save the best model so far, in addition, save the latest 
                # model into a separate file with the iteration number for 
                # external eval
                if numpy.mod(uidx, saveFreq) == 0:
                    if not os.path.exists(saveto_dir):
                        os.mkdir(saveto_dir)
                    print 'Saving the best model...',
                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(saveto, history_errs=history_errs, uidx=uidx, **params)
                    pkl.dump(options, open('%s.pkl' % saveto, 'wb'))
                    print 'Done'

                    # save with uidx
                    if not overwrite:
                        print 'Saving the model at iteration {}...'.format(uidx),
                        saveto_uidx = '{}.iter{}.npz'.format(
                            os.path.splitext(saveto)[0], uidx)
                        numpy.savez(saveto_uidx, history_errs=history_errs,
                                    uidx=uidx, **unzip(tparams))
                        print 'Done'


                # generate some samples with the model and display them
                if numpy.mod(uidx, sampleFreq) == 0:
                    # FIXME: random selection?
                    for jj in xrange(numpy.minimum(5, x.shape[1])):
                        sample, score = gen_sample(tparams, f_init, f_next,
                                                   x[:, jj][:, None],
                                                   keyword[:, jj][:, None],
                                                   keyword_mask[:, jj][:, None], 
                                                   options, trng=trng, k=1,
                                                   maxlen=maxlen,
                                                   stochastic=stochastic,
                                                   argmax=argmax)
                        print 'Source ', jj, ': ',
                        for vv in x[:, jj]:
                            if vv == 0:
                                break
                            if vv in worddicts_r[0]:
                                print worddicts_r[0][vv],
                            else:
                                print 'UNK',
                        print
                        print 'Truth ', jj, ' : ',
                        for vv in y[:, jj]:
                            if vv == 0:
                                break
                            if vv in worddicts_r[1]:
                                print worddicts_r[1][vv],
                            else:
                                print 'UNK',
                        print
                        print 'Sample ', jj, ': ',
                        if stochastic:
                            ss = sample
                        else:
                            score = score / numpy.array([len(s) for s in sample])
                            ss = sample[score.argmin()]
                        for vv in ss:
                            if vv == 0:
                                break
                            if vv in worddicts_r[1]:
                                print worddicts_r[1][vv],
                            else:
                                print 'UNK',
                        print

                # validate model on validation set and early stop if necessary
                if numpy.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    valid_errs = pred_probs(f_log_probs, prepare_data,
                                            options, valid, verbose = True)
                    valid_err = valid_errs.mean()
                    history_errs.append(valid_err)

                    if uidx == 0 or valid_err <= numpy.array(history_errs).min():
                        best_p = unzip(tparams)
                        bad_counter = 0
                    if len(history_errs) > patience and valid_err >= \
                            numpy.array(history_errs)[:-patience].min():
                        bad_counter += 1
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break

                    if numpy.isnan(valid_err):
                        ipdb.set_trace()

                    print 'Valid ', valid_err

                # finish after this many updates
                if uidx >= finish_after:
                    print 'Finishing after %d iterations!' % uidx
                    estop = True
                    break

                uidx += 1

            print 'Seen %d samples' % n_samples

            if estop:
                break
             


        if best_p is not None:
            zipp(best_p, tparams)

        use_noise.set_value(0.)
        valid_err = pred_probs(f_log_probs, prepare_data,
                               options, valid).mean()

        print 'Valid ', valid_err

        params = copy.copy(best_p)
        numpy.savez(saveto, zipped_params=best_p,
                    history_errs=history_errs,
                    uidx=uidx,
                    **params)
    else:
        fout = open(test_datasets[1], 'w')
        for keyword, imageids in test:
            keyword, keyword_mask = prepare_data(None, None, keyword, phase = 'test')
            previous_source_seq = numpy.zeros((2, 1))
            previous_source_seq = numpy.zeros((2, 1)).astype('int64')
            previous_source_seq[0, 0] = worddicts[0]['START']
            previous_source_seq[1, 0] = worddicts[0]['STOP']

            pred_target = numpy.zeros((3, 1)).astype('int64')
            pred_target[0, 0] = worddicts[0]['START']
            pred_target[1, 0] = worddicts[0]['STOP']
            pred_target[2, 0] = worddicts[0]['eos']

            
            for line in range(keyword.shape[2]):
                previous_source_seq = prepare_test_source(previous_source_seq, 
                                                      worddicts[0], line, 
                                                      pred_target) 
                # print previous_source_seq
                sample, score = gen_sample(tparams, f_init, f_next,
                                           previous_source_seq, 
                                           keyword[:, :, line], 
                                           keyword_mask[:, :, line],
                                           options, trng=trng, k=1,
                                           maxlen=30,
                                           stochastic=stochastic,
                                           argmax=argmax)
                if stochastic:
                    ss = sample
                else:
                    score = score / numpy.array([len(s) for s in sample])
                    ss = sample[score.argmin()]
                pred_target = numpy.zeros((len(ss), 1))
                for i in range(len(ss)):
                    pred_target[i, 0] = ss[i]
                # for vv in ss:
                #     if vv == 0:
                #         break
                #     if vv in worddicts_r[1]:
                #         print worddicts_r[1][vv],
                #     else:
                #         print 'UNK',
            target_seq = []
            for i in range(1, previous_source_seq.shape[0] - 2):
                if previous_source_seq[i, 0] in worddicts_r[1]:
                    target_seq.append(
                            worddicts_r[1][previous_source_seq[i, 0]])
                else:
                    target_seq.append('UNK')
            target_seq.append('，')
            for i in range(1, pred_target.shape[0] - 2):
                if pred_target[i, 0] in worddicts_r[1]:
                    target_seq.append(worddicts_r[1][pred_target[i, 0]])
                else:
                    target_seq.append('UNK')
            seq = ' '.join(target_seq)

            fout.write('{}@'.format(imageids[0]))
            fout.write(seq)
            fout.write('\n')
        fout.close()
            
    return 'train' == mode and valid_err or 0


if __name__ == '__main__':
    pass
