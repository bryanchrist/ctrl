from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
import transformer
import argparse
import pdb
import sys
import re
from collections import Counter
from tensorflow.python import debug as tf_debug
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import embedding_ops
import fastBPE
import platform
from control_codes import CONTROL_CODES
from tensorflow.keras import layers

use_py3 = platform.python_version()[0] == '3'

parser = argparse.ArgumentParser(description='TensorFlow code for generating from CTRL')
parser.add_argument('--model_dir', type=str, required=True,
                                        help='location of model checkpoint')
parser.add_argument('--seed', type=int, default=1337,
                                        help='random seed for TensorFlow, numpy and PythonHash')
parser.add_argument('--generate_num', type=int, default=256,
                                        help='number of tokens to generate')
parser.add_argument('--temperature', type=float, default=0,
                                        help='temperature for sampling distribution; 0 means greedy')
parser.add_argument('--nucleus', type=float, default=0.,
                                        help='cumulative probability cutoff for nucleus sampling; 0 means no nucleus sampling')
parser.add_argument('--topk', type=int, default=0,
                                        help='topk value for sampling from the softmax distribution ; 0 means no topk preferred')
parser.add_argument('--penalty', type=float, default=1.2,
                                        help='repetition penalty for greedy sampling')
parser.add_argument('--print_once', action='store_true',
                                        help='the completion is printed only at the end; not every word')
parser.add_argument('--topn', type=int, default=0,
                                        help='print top-n candidates during generations; defaults to 0 which is no printing')

args = parser.parse_args()
tf.random.set_seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)

# load the vocabulary from file
vocab = open('vocab').read().decode(encoding='utf-8').split('\n') if not use_py3 else open('vocab', encoding='utf-8').read().split('\n')
vocab = list(map(lambda x: x.split(' ')[0], vocab)) + ['<unk>'] + ['\n']
print ('{} unique words'.format(len(vocab)))

# length of the vocabulary
vocab_size = len(vocab)

# define the numericalization map
# idx2word maps the numericalized ID to the word
# word2idx maps the word to the numericalized ID
word2idx = {u:i for i, u in enumerate(vocab)}
idx2word = np.array(vocab)



# sequence length to use for the transformer
# the model is trained with a seq_length of 512
# so, any value <= 512 should work
seq_length = min(args.generate_num, 256)




# the dimension of the transformer
embedding_dim = 1280


# Now, we begin defining the model
# we defer the transformer definition to transformer.py
# here, we only define the tied softmax layer
# this layer ties the softmax weights to the input embeddings
class TiedEmbeddingSoftmax(layers.Layer):
    def __init__(self, tied_embedding_layer, **kwargs):
        super(TiedEmbeddingSoftmax, self).__init__(**kwargs)
        self.w = tied_embedding_layer.embeddings

    def call(self, inputs, embed=True):
        if embed:
            return tf.nn.embedding_lookup(self.w, inputs)
        else:
            return tf.matmul(inputs, tf.transpose(self.w))

# input for the keras model
tokens = tf.keras.layers.Input(shape=(seq_length,), dtype='int32')

# instantiates a tied softmax class
tied_embedding_softmax = TiedEmbeddingSoftmax()

# embedded tokens, before passing it to the transformer
embedded = tied_embedding_softmax(tokens, embed=True)

# the activations after passing it from the transformer
# for some odd reason, TPUs don't play well with specifying the arguments of the Encoder() function
# so you have to leave them at their defaults
transformed = transformer.Encoder()(embedded, training=False)


# pass the activations from our tiedsoftmax class
# this time with embed=False denoting that we are doing the softmax operation
# and not a lookup
logits = tied_embedding_softmax(transformed, embed=False)


# finally, define the Keras model with inputs as tokens and outputs as the logits we just computed
model = tf.keras.Model(inputs=tokens, outputs=logits)


# the loss function is a simple categorical crossentropy between the logits and the labels
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# the optimizer is not used since this code only supports inference
# however, to compile the model, we still define it
import tensorflow.compat.v1 as tf

# Disable eager execution
tf.compat.v1.disable_eager_execution()

# however, to compile the model, we still define it
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

# # Define placeholder for batch size
# batch_size = tf.shape(tokens)[0]

# # Placeholder for labels
# labels = tf.compat.v1.placeholder(shape=(batch_size, seq_length), dtype=tf.int32)  # Replace `batch_size` with the actual batch size

# # Calculate loss
# loss_value = loss(labels, logits)

# with tf.GradientTape() as tape:
#     logits = model(tokens)
#     loss_value = loss(labels, logits)

# grads = tape.gradient(loss_value, model.trainable_variables)
# clipped_grads, _ = tf.clip_by_global_norm(grads, 0.25)
# train_op = optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))



# compile the model with the optimizer and loss            
model.compile(optimizer=optimizer, loss=loss)
print(model.summary())


# IMPORTANT
# this is where the saved model is presented to the code
# the model directory should have the model checkpoint and
# a checkpoint file
run_config = tf.estimator.RunConfig(model_dir=args.model_dir)



# this converts the Keras model to a TensorFlow estimator
# this step is critical
# remember to patch the TF 1.14 file before running the code, else you're going to see errors here
def model_fn(features, labels, mode):
    # Define the model operations using the features input
    tokens = features['input_1']
    embedded = tied_embedding_softmax(tokens, embed=True)
    transformed = transformer.Encoder()(embedded, training=False)
    logits = tied_embedding_softmax(transformed, embed=False)

    # Create the model spec based on the mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        loss_value = loss(labels, logits)
        train_op = optimizer.minimize(loss_value, global_step=tf.train.get_global_step())

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode, loss=loss_value, train_op=train_op)
        else:  # mode == tf.estimator.ModeKeys.EVAL
            eval_metric_ops = {
                'accuracy': tf.metrics.accuracy(labels, tf.argmax(logits, axis=-1)),
            }
            return tf.estimator.EstimatorSpec(mode, loss=loss_value, eval_metric_ops=eval_metric_ops)

# Create the estimator model using the model_fn
estimator_model = tf.estimator.Estimator(model_fn=model_fn, config=run_config)

# we now create a serving function from this estimator
# this enables us to load the model once and easily query it multiple times
# Define the serving input function
def serving_input_receiver_fn():
    inputs = {'input_1': tf.placeholder(tf.int32, [1, seq_length])}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

# Load the model using export_saved_model with serving_input_receiver_fn
estimator_model.export_saved_model(args.model_dir, serving_input_receiver_fn)
predict_fn = tf.saved_model.load(args.model_dir)


# almost there, we now take the user prompt and tokenize with BPE
# load BPE codes
bpe = fastBPE.fastBPE('codes', 'vocab')

temperature = args.temperature
nucleusprob = args.nucleus
penalty = args.penalty
topk = args.topk

while True:
    prompt = input('ENTER PROMPT: ')
    prompt = prompt.split('\\n') # split on newlines if provided

    # tokenize provided prompt
    split_prompt = ' \n '.join(bpe.apply(prompt))
    split_prompt = split_prompt.split(' ')

    if not any(split_prompt[0] == x for x in CONTROL_CODES.keys()):
        print("WARNING! You are not starting your generation from a control code so you won't get good results")
    text = [word2idx[i] for i in split_prompt]

    # pad with 0s and create a mini-batch of 2 (arbitrary, for ease of code)
    padded_text = text + [0] * (args.generate_num - len(text))
    tokens_generated = np.tile(padded_text, (1,1))
    try:
        for token in range(len(text)-1, args.generate_num-1):
          # get the logits from the prediction function
          # the logic here is a bit convoluted because we are allowing generation past 512 tokens
          # this is done by sliding the window over (past 512 tokens) and continuing prediction
          # I'm sure this can be simplified (TODO)
          if token <= seq_length:
            prompt_logits = predict_fn({'input_1':tokens_generated[:, :seq_length]})['tied_embedding_softmax'].squeeze() / (temperature if temperature>0 else 1.)
            _token = token if token < seq_length else -1
          else:
            _token = -1
            end = token + 1
            start = token - seq_length + 2
            prompt_logits = predict_fn({'input_1':np.hstack((tokens_generated[:,0:1], tokens_generated[:,start:end]))})['tied_embedding_softmax'].squeeze() / (temperature if temperature>0 else 1.)


          # if penalty (for repetition) is non-zero,
          # discount the logits from already generated tokens
          if penalty>0:
              penalized_so_far = set()
              for _ in range(token+1):
                 generated_token = tokens_generated[0][_]
                 # don't penalize newlines
                 # you could also choose not to penalize frequent words
                 # (which incidentally are sorted in the vocab file)
                 # but I don't do that
                 # if it prints too many new lines instead of continuing generating text,
                 # you might want to comment this out
                 if idx2word[generated_token] == '\n':
                     continue
                 if generated_token in penalized_so_far:
                     continue
                 penalized_so_far.add(generated_token)
                 prompt_logits[_token][generated_token] /= penalty

          # disallow some tokens
          prompt_logits[_token][word2idx['<unk>']] = -1e8

          # sometimes, when generating from reddit,
          # it tries to generate the Score (reddit Karma) immediately after generating the Title:
          # to disallow this, we can just prevent it from generating Score
          prompt_logits[_token][word2idx['Sco@@']] = -1e8


          # compute probabilities from logits
          prompt_probs = np.exp(prompt_logits[_token])
          prompt_probs = prompt_probs / sum(prompt_probs)
          pruned_list = np.argsort(prompt_probs)[::-1]
          # if you are using nucleus prob, then compute the nucleus probability size
          if nucleusprob > 0.:
            minimum_topk = 1
            nucleus = max(np.where(np.cumsum(np.sort(prompt_probs)[::-1])>nucleusprob)[0][0], minimum_topk)
          elif topk > 0:
            # we are over-loading notation here
            # if you choose to specify a topk instead of a nucleus,
            # we will hardcode the nucleus to be just that
            nucleus = topk
          else:
            # if you specify neither nucleus or topk,
            # then we will use the whole list
            nucleus = len(pruned_list)
            
          pruned_list = pruned_list[:nucleus]  
          # if you want to disallow more complex tokens, you can do so here
          # for instance, if you want to disallow anything with the phrase `http`,
          # you can delete theme from the pruned_list
          # you can comment this out, I'm keeping it in for demonstration purpose
          tokens_to_disallow = []
          for _ in range(len(pruned_list)):
              if 'http' in idx2word[pruned_list[_]]:
                  tokens_to_disallow.append(_)
          pruned_list = np.delete(pruned_list, tokens_to_disallow)

          if args.topn > 0 :
            print('TOPN :: top-n alternatives:', [idx2word[_] for _ in pruned_list[:args.topn]])

          # if temperature is 0
          # just pick the first (most probable) token
          if temperature==0:
              idx = pruned_list[0]
          else:
            # else,
            # sample from the pruned_list with the logits
            chosen_idx = int(tf.random.categorical(np.expand_dims(prompt_logits[_token][pruned_list],0), num_samples=1).numpy())
            idx = pruned_list[chosen_idx]

          if args.topn > 0 :
            print('TOPN :: chosen word:', idx2word[idx])

          # assign the token for generation
          tokens_generated[0][token+1] = idx

          # clear screen if you want to
          # os.system("clear")


          tokens_generated_so_far = ' '.join([idx2word[c] for c in tokens_generated[0].squeeze()[:token+2]])
          tokens_generated_so_far = re.sub('(@@ )', '', string=tokens_generated_so_far)
          tokens_generated_so_far = re.sub('(@@ ?$)', '', string=tokens_generated_so_far)              

          if not args.print_once:
            print('---------------------------------------')
            print(tokens_generated_so_far)
            print()
        print('---------------------------------------')            
        print(tokens_generated_so_far)
        print()

    except KeyboardInterrupt: #Exception as e:
        print('Continuing')
            
