import os
import numpy as np
import tensorflow as tf



dataset_text = open('harry.txt', 'rb').read().decode(encoding='utf-8')
# print(dataset_text)

    # ici on prendre le nombre de caratere unique dans le jeu le text upper case , lower case et pontuation
    # la function set sert a prendre le element unique du text
vocab = sorted(set(dataset_text))
print ('{} unique characters'.format(len(vocab)))

# ici on crée un mapping des caratere unique a quel idex ce trouve le carataire
char2idx = {char:index for index, char in enumerate(vocab)}


# transforme le array de unique carataire en numpy array
idx2char = np.array(vocab)

# transforme chaque lettre en avec l'index qui lui apartiet dans le tableau
# text_as_int = dataset_text mais avec des chiffre aulieu de lettre chiffre qui son les index du tablea char2idx
text_as_int = np.array([char2idx[char] for char in dataset_text])


print ('{} ---- characters mapped to int ---- > {}'.format(repr(dataset_text[:13]), text_as_int[:13]))




# Example: if our text is "Hello" and seq_len = 4

len(dataset_text)

# Calculate the number of examples per epoch assuming a sequence length of 100 characters
seq_length = 300

# We will divide the dataset into a squence of characters with seq_length
# chaque epoch va contenir un nombre de phrase ex si on a
# 20000 carataire dans nom histoire et on prendre des sequance de 200 carataire alors nous allons avoir 20000/200 exemple a donner a nom model
examples_per_epoch = len(dataset_text)//seq_length
print(examples_per_epoch)
# va crée un tensor Tensors are multi-dimensional arrays
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)


# Try a different number, let's say first 200
for i in char_dataset.take(10):
  print(idx2char[i.numpy()], i.numpy())




# The batch method lets us easily convert these individual characters to sequences of the desired size.
# prend notre dataset char_dataset et le met en array de array avec le nb de carataire voulu seq_length
# on fait un +1 pour avoir 201 carartaire pour que la function split_input_target return en input et output 200 carataire
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
# Try 10

for item in sequences.take(50):
  print(repr(''.join(idx2char[item.numpy()])))


 # For each sequence, duplicate and shift it to form the input and target text by using the `map` method to apply a simple function to each batch:
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)


# The output will be the same as the input but shifted by one character ex:
# Input: "Hell"
# Output: "ello"
for input_example, target_example in dataset.take(10):
    print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
    # print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))



# mets 64 par array
BATCH_SIZE = 64
# bouge chaque element de 10000
BUFFER_SIZE = 10000
# pour melanger les phrase
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

for input_example, target_example in dataset.take(10):
    for i in input_example:
        print ('Input data: ', repr(''.join(idx2char[i.numpy()])))

# Length of the vocabulary in chars
vocab_size = len(vocab)
# The embedding dimension
embedding_dim = 10
# Number of RNN units
rnn_units = 10



def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    #   La couche d'entrée. Une table de recherche entraînable qui mappera les nombres de chaque caractère à un vecteur avec
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),

      tf.keras.layers.GRU(rnn_units,
                          return_sequences=True,
                          stateful=True,
                          recurrent_initializer='glorot_uniform'),

      tf.keras.layers.GRU(rnn_units,
                          return_sequences=True,
                          stateful=True,
                          recurrent_initializer='glorot_uniform'),

    tf.keras.layers.Dense(vocab_size)
  ])
  return model



model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)


for input_example_batch, target_example_batch in dataset.take(10):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")


model.summary()


sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()

# lorsq que l'input est ca
print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
# le model non entrainer va donner ça
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))



def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())


model.compile(optimizer='adam', loss=loss)

# save le mkodel pendant le tranning a chaque epoch
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


EPOCHS=25


history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

tf.train.latest_checkpoint(checkpoint_dir)


model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

model.summary()


def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 1000

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the word returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))



print(generate_text(model, start_string=u"Harry: "))
