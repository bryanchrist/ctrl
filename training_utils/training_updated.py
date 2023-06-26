import tensorflow as tf
import os
import numpy as np
import transformer
import argparse
import pdb
import re
from collections import Counter
import fastBPE
import platform

use_py3 = platform.python_version()[0] == '3'

parser = argparse.ArgumentParser(description='TensorFlow code for generating from CTRL')
parser.add_argument('--model_dir', type=str, required=True,
                    help='location of model checkpoint')
parser.add_argument('--seed', type=int, default=1337,
                    help='random seed for TensorFlow, numpy and PythonHash')
parser.add_argument('--sequence_len', type=int, default=256,
                    help='sequence len of model being fine-tuned (must match also the TFRecords)')
parser.add_argument('--iterations', type=int, default=1000,
                    help='random seed for TensorFlow, numpy and PythonHash')

args = parser.parse_args()
tf.random.set_seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)

# Load the vocabulary from file
vocab = open('../vocab').read().decode(encoding='utf-8').split('\n') if not use_py3 else open('../vocab', encoding='utf-8').read().split('\n')
vocab = list(map(lambda x: x.split(' ')[0], vocab)) + ['<unk>'] + ['\n']
print('{} unique words'.format(len(vocab)))

# Length of the vocabulary
vocab_size = len(vocab)

# Define the numericalization map
# idx2word maps the numericalized ID to the word
# word2idx maps the word to the numericalized ID
word2idx = {u: i for i, u in enumerate(vocab)}
idx2word = np.array(vocab)

# Sequence length to use for the transformer
# Must match the model being fine-tuned
seq_length = args.sequence_len

def input_fn(params=None):
    print('READING!', params)
    dataset = tf.data.Dataset.list_files('./*.tfrecords', shuffle=True)

    tf_data = tf.data.TFRecordDataset(dataset)
    myfeatures = {
        'input': tf.io.FixedLenFeature([256], tf.int64),
        'output': tf.io.FixedLenFeature([256], tf.int64)
    }

    def _parse_text_function(example_proto):
        blah = tf.io.parse_single_example(example_proto, myfeatures)
        return blah['input'], blah['output']

    train_data = tf_data.map(_parse_text_function).batch(params['batch_size'], drop_remainder=True).repeat().shuffle(10000)
    
    return train_data

# The dimension of the transformer
embedding_dim = 1280

# Now, we begin defining the model
# We defer the transformer definition to transformer.py
# Here, we only define the tied softmax layer
# This layer ties the softmax weights to the input embeddings
class TiedEmbeddingSoftmax(tf.keras.layers.Layer):

    def __init__(self, vocab_size=vocab_size, embedding_size=embedding_dim, **kwargs):
        super(TiedEmbeddingSoftmax, self).__init__()
        self.w = self.add_weight(name='w', shape=(vocab_size, embedding_size),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='b', shape=(vocab_size,),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs, embed=True):
        if embed:
            dtype = inputs.dtype
            if dtype != 'int32' and dtype != 'int64':
                inputs = tf.cast(inputs, 'int32')
            return tf.nn.embedding_lookup(self.w, inputs)
        else:
            return tf.matmul(inputs, tf.transpose(self.w)) + self.b

# Input for the Keras model
tokens = tf.keras.layers.Input(shape=(seq_length,), dtype='int32')

# Instantiates a tied softmax class
tied_embedding_softmax = TiedEmbeddingSoftmax()

# Embedded tokens, before passing it to the transformer
embedded = tied_embedding_softmax(tokens, embed=True)

# The activations after passing it from the transformer
# For some odd reason, TPUs don't play well with specifying the arguments of the Encoder() function
# So you have to leave them at their defaults
transformed = transformer.Encoder()(embedded, training=False)

# Pass the activations from our tied softmax class
# This time with embed=False denoting that we are doing the softmax operation
# and not a lookup
logits = tied_embedding_softmax(transformed, embed=False)

# Finally, define the Keras model with inputs as tokens and outputs as the logits we just computed
model = tf.keras.Model(inputs=tokens, outputs=logits)

# The loss function is a simple categorical crossentropy between the logits and the labels
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# The optimizer is not used since this code only supports inference
# However, to compile the model, we still define it
optimizer = tf.keras.optimizers.Adagrad(learning_rate=1e-2)

# Compile the model with the optimizer and loss
model.compile(optimizer=optimizer, loss=loss)
print(model.summary())

# This is where the saved model is presented to the code
# The model directory should have the model checkpoint and a checkpoint file
run_config = tf.estimator.RunConfig(model_dir=args.model_dir)

# Convert the Keras model to a TensorFlow estimator
estimator_model = tf.keras.estimator.model_to_estimator(keras_model=model, config=run_config)

# Train the model
estimator_model.train(input_fn=input_fn, steps=args.iterations)
