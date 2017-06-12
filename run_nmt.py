#!/usr/bin/env python

import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help = 'model to run')
    parser.add_argument('mode', help = 'Running mode')
    parser.add_argument('--train_source', action = 'store', 
            dest = 'train_source', 
            help = 'train source',
            default = 'image_poetry_pair_source_train')
    parser.add_argument('--train_target', action = 'store', 
            dest = 'train_target', 
            help = 'train target',
            default = 'image_poetry_pair_target_train')
    parser.add_argument('--val_source', action = 'store', 
            dest = 'val_source', 
            help = 'validation source',
            default = 'image_poetry_pair_source_val')
    parser.add_argument('--val_target', action = 'store', 
            dest = 'val_target', 
            help = 'validation target',
            default = 'image_poetry_pair_target_val')
    parser.add_argument('--test_source', action = 'store', 
            dest = 'test_source', 
            help = 'test source',
            default = 'test_imageids_with_memory')
    parser.add_argument('--test_target', action = 'store', 
            dest = 'test_target', 
            help = 'test target',
            default = '')
    parser.add_argument('--conv5_dir', action = 'store', 
            dest = 'conv5_dir', 
            help = 'conv5 feature',
            default = 'clean_features_vgg_vd19_conv5_npy_float32')
    parser.add_argument('--fc7_dir', action = 'store', 
            dest = 'fc7_dir', 
            help = 'fc7 feature',
            default = 'clean_features_vgg_vd19_fc7_npy_float32')
    parser.add_argument('--source_dict', action = 'store', 
            dest = 'source_dict', 
            help = 'source dictionary',
            default = 'new_corpus.pkl')
    parser.add_argument('--target_dict', action = 'store', 
            dest = 'target_dict', 
            help = 'target dictionary',
            default = 'new_corpus.pkl')
    parser.add_argument('--train_reference', action = 'store', 
            dest = 'train_reference', 
            default = 'image_poetry_pair_image_memory_train', 
            help = 'train reference')
    parser.add_argument('--val_reference', action = 'store', 
            dest = 'val_reference', 
            default = 'image_poetry_pair_image_memory_val', 
            help = 'val reference')
    parser.add_argument('--train_keyword', action = 'store', 
            dest = 'train_keyword', 
            default = 'train_keyword_seq', 
            help = 'train keyword')
    parser.add_argument('--val_keyword', action = 'store', 
            dest = 'val_keyword', 
            default = 'val_keyword_seq', 
            help = 'val keyword')
    parser.add_argument('--train_sentiment', action = 'store', 
            dest = 'train_sentiment', 
            default = 'image_poetry_pair_image_senti_memory_train', 
            help = 'train sentiment')
    parser.add_argument('--val_sentiment', action = 'store', 
            dest = 'val_sentiment', 
            default = 'image_poetry_pair_image_senti_memory_val', 
            help = 'val sentiment')
    parser.add_argument('--reference_dictionary', action = 'store', 
            dest = 'reference_dictionary', 
            help = 'reference dictionary',
            default = 'memory_corpus.pkl')
    parser.add_argument('--sentiment_dictionary', action = 'store', 
            dest = 'sentiment_dictionary', 
            help = 'sentiment dictionary',
            default = 'senti_dict.pkl')
    parser.add_argument('--pos_sentiment_mem', action = 'store', 
            dest = 'pos_sentiment_mem', 
            help = 'positive sentiment memory',
            default = 'pos_sentiDataset.pkl')
    parser.add_argument('--neg_sentiment_mem', action = 'store', 
            dest = 'neg_sentiment_mem', 
            help = 'negative sentiment memory',
            default = 'neg_sentiDataset.pkl')
    parser.add_argument('--saveto', action = 'store', 
            dest = 'saveto', 
            help = 'where to save model',
            default = 'image_poem_mem.npz')
    parser.add_argument('--batch_size', action = 'store', 
            dest = 'batch_size', type = int, default = 128, 
            help = 'batch_size')
    parser.add_argument('--val_batch_size', action = 'store', 
            dest = 'val_batch_size', type = int, default = 128, 
            help = 'val_batch_size')
    parser.add_argument('--reload', action = 'store_true', 
            dest = 'reload_flag', help = 'Where to reload parameters')
    parser.add_argument('--reload_iter', action = 'store', 
            dest = 'reload_iter', default = -1, type = int,
            help = 'Where to reload parameters')
    parser.add_argument('--stochastic', action = 'store_false', 
            dest = 'stochastic', default =True,
            help = 'Use beam search or stochastic while generating sample')
    parser.add_argument('--argmax', action = 'store_true', 
            dest = 'argmax', default = False,
            help = 'Use beam search or argmax while generating sample')
    parser.add_argument('--basedir', action = 'store', 
            dest = 'basedir', type = str, 
            default = '', 
            help = 'base directory')
    parser.add_argument('--n_words_src', type = int,
            action = 'store', dest = 'n_words_src', default = 6000, 
            help = 'number of words in source')
    parser.add_argument('--n_words', type = int,
            action = 'store', dest = 'n_words', default = 6000, 
            help = 'number of words in target')
    parser.add_argument('--n_words_reference', type = int,
            action = 'store', dest = 'n_words_reference', default = 4000, 
            help = 'number of words in reference')
    parser.add_argument('--maxlen', type = int,
            action = 'store', dest = 'maxlen', default = 30, 
            help = 'max length of generated sequence')
    parser.add_argument('--validSize', type = int,
            action = 'store', dest = 'validSize', default = 1024, 
            help = 'valid size')
    parser.add_argument('--valid_batch_size', type = int,
            action = 'store', dest = 'valid_batch_size', default = 128, 
            help = 'valid batch size')
    parser.add_argument('--decay_c', type = float,
            action = 'store', dest = 'decay_c', default = 0., 
            help = 'decay ratio of c')
    parser.add_argument('--alpha_c', type = float,
            action = 'store', dest = 'alpha_c', default = 0., 
            help = 'regularizer ratio of c')
    parser.add_argument('--clip_c', type = float,
            action = 'store', dest = 'clip_c', default = 1.,
            help = 'threshold to clip c')
    parser.add_argument('--optimizer', action = 'store', 
            dest = 'optimizer', type = str, 
            default = 'adadelta', 
            help = 'optimizer')
    parser.add_argument('--validFreq', type = int,
            action = 'store', dest = 'validFreq', default = 1000, 
            help = 'validation frequency')
    parser.add_argument('--saveFreq', type = int,
            action = 'store', dest = 'saveFreq', default = 1000, 
            help = 'save frequency')
    parser.add_argument('--sampleFreq', type = int,
            action = 'store', dest = 'sampleFreq', default = 1000, 
            help = 'sample frequency')
    parser.add_argument('--dispFreq', type = int,
            action = 'store', dest = 'dispFreq', default = 100, 
            help = 'display frequency')
    parser.add_argument('--max_epochs', type = int,
            action = 'store', dest = 'max_epochs', default = 1000,
            help = 'max epochs to run')
    parser.add_argument('--lrate', type = float,
            action = 'store', dest = 'lrate', default = 0.0001,
            help = 'Learning rate')
    parser.add_argument('--overwrite', action = 'store_true', 
            dest = 'overwrite', help = 'Whether overwrite while saving')
    parser.add_argument('--patience', type = int,
            action = 'store', dest = 'patience', default = 10,
            help = 'patience fo early stop')
    parser.add_argument('--finish_after', type = int,
            action = 'store', dest = 'finish_after', default = 10000000,
            help = 'iteration to finish')
    parser.add_argument('--fc7_size', type = int,
            action = 'store', dest = 'fc7_size', default = 4096,
            help = 'fc7 image size')
    parser.add_argument('--dim', type = int,
            action = 'store', dest = 'dim', default = 512,
            help = 'dimesion of gru')
    parser.add_argument('--dim_word', type = int,
            action = 'store', dest = 'dim_word', default = 512,
            help = 'dimesion of word embeddings')
    parser.add_argument('--dim_memory_input', type = int,
            action = 'store', dest = 'dim_memory_input', default = 512,
            help = 'dimesion of input memory')
    parser.add_argument('--dim_memory_output', type = int,
            action = 'store', dest = 'dim_memory_output', default = 512,
            help = 'dimesion of output memory')
    parser.add_argument('--encoder', type = str,
            action = 'store', dest = 'encoder', default = 'gru', 
            help = 'encoder to use')
    parser.add_argument('--decoder', type = str,
            action = 'store', dest = 'decoder', default = 'gru_cond', 
            help = 'decoder to use')
    parser.add_argument('--dimi', type = int,
            action = 'store', dest = 'dimi', default = 512,
            help = 'dimesion of images features')
    parser.add_argument('--use_dropout', action = 'store_true', 
            default = False, 
            help = 'If use droput')
    parser.add_argument('--num_data_load_workers', type = int, 
            action = 'store', dest = 'num_data_load_workers', default = 8,
            help = 'Number of data loading worker')
    parser.add_argument('--data_queue_size', type = int, 
            action = 'store', dest = 'data_queue_size', default = 16, 
            help = 'Size of data queue')
    parser.add_argument('--train_skip', type = int, 
            action = 'store', dest = 'train_skip', default = 0, 
            help = 'Line to skip in training set')
    parser.add_argument('--valid_skip', type = int, 
            action = 'store', dest = 'valid_skip', default = 0, 
            help = 'Line to skip in validation set')
    parser.add_argument('--level', type = str, 
            action = 'store', dest = 'level', default = 'level.pkl', 
            help = 'Level file')
    parser.add_argument('--tone', type = str, 
            action = 'store', dest = 'tone', default = 'tone.pkl', 
            help = 'tone file')
    parser.add_argument('--phase', type = str, choices = ['image', 'poetry'],
            action = 'store', dest = 'phase', default = 'image', 
            help = 'Training phase')
    return parser

def main():
    parser = argparser()
    args = parser.parse_args()
    params = dict()
    for arg in vars(args):
        params[arg] = getattr(args, arg)
    print params
    print 'from %s import run' % params['model']
    exec 'from %s import run' % params['model']
    validerr = run(params)
    return validerr

if __name__ == '__main__':
    main()
