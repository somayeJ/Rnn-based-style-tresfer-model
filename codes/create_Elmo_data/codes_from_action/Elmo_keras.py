# Nazanin , model with Imdb
import keras
import os
import sys
from allennlp.commands.elmo import ElmoEmbedder
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Activation, Dropout


# Load all files from a directory into dictionaries
def load_directory_data(directory, label):
    data = []
    for file_path in os.listdir(directory):
        with open(os.path.join(directory, file_path), "r") as f:
            data.append({"text": f.read().replace("<br />", " "), "label": label})
    return data

# Load the positive and negative examples from the dataset
def load_dataset(directory):
    pos_data = load_directory_data(os.path.join(directory, "pos"), 1)
    neg_data = load_directory_data(os.path.join(directory, "neg"), 0)
    return pos_data+neg_data

# Download and process the IMDB dataset
def download_and_load_datasets(force_download=False):
    '''

    :param force_download:
    :return: data1, data2
    data1: list of dictionaries corresponding to each review consisting of the three components as follows:
     {'tokens': 'text', label: '1 or 0' , 'tokens':[...]}
    '''
    dataset = keras.utils.get_file(
      fname="aclImdb.tar.gz",
      origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
      extract=True)

    train_data = load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", "train"))
    test_data = load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", "test"))

    return train_data, test_data

train_data, test_data = download_and_load_datasets()

random.shuffle(train_data)
random.shuffle(test_data)


#Tokenize text. Note, it would be better to first split it into sentences.
def tokenize_text(documents, max_tokens):
    for document in documents:
        document['tokens'] = keras.preprocessing.text.text_to_word_sequence(document['text'], lower=False)
        document['tokens'] = document['tokens'][0:max_tokens]

max_tokens = 100
tokenize_text(train_data, max_tokens)
tokenize_text(test_data, max_tokens)

print(train_data, type(train_data))

# Lookup the ELMo embeddings for all documents (all sentences) in our dataset. Store those
# in a numpy matrix so that we must compute the ELMo embeddings only once.
def create_elmo_embeddings(elmo, documents, max_sentences = 1000):
    num_sentences = min(max_sentences, len(documents)) if max_sentences > 0 else len(documents)
    print("\n\n:: Lookup of "+str(num_sentences)+" ELMo representations. This takes a while ::")
    embeddings = []
    labels = []
    tokens = [document['tokens'] for document in documents]

    documentIdx = 0
    for elmo_embedding in elmo.embed_sentences(tokens):
        document = documents[documentIdx]
        # Average the 3 layers returned from ELMo
        avg_elmo_embedding = np.average(elmo_embedding, axis=0)

        embeddings.append(avg_elmo_embedding)
        labels.append(document['label'])

        # Some progress info
        documentIdx += 1
        percent = 100.0 * documentIdx / num_sentences
        line = '[{0}{1}]'.format('=' * int(percent / 2), ' ' * (50 - int(percent / 2)))
        status = '\r{0:3.0f}%{1} {2:3d}/{3:3d} sentences'
        sys.stdout.write(status.format(percent, line, documentIdx, num_sentences))

        if max_sentences > 0 and documentIdx >= max_sentences:
            break

    return embeddings, labels


elmo = ElmoEmbedder() #Set cuda_device to the ID of your GPU if you have one
#elmo = ElmoEmbedder(cuda_device=0) #Set cuda_device to the ID of your GPU if you have one
train_x, train_y = create_elmo_embeddings(elmo, train_data, 1000)
test_x, test_y  = create_elmo_embeddings(elmo, test_data, 1000)



# :: Pad the x matrix to uniform length ::
def pad_x_matrix(x_matrix):
    for sentenceIdx in range(len(x_matrix)):
        sent = x_matrix[sentenceIdx]
        sentence_vec = np.array(sent, dtype=np.float32)
        padding_length = max_tokens - sentence_vec.shape[0]
        if padding_length > 0:
            x_matrix[sentenceIdx] = np.append(sent, np.zeros((padding_length, sentence_vec.shape[1])), axis=0)

    matrix = np.array(x_matrix, dtype=np.float32)
    return matrix

train_x = pad_x_matrix(train_x)
train_y = np.array(train_y)

test_x = pad_x_matrix(test_x)
test_y = np.array(test_y)

print("Shape Train X:", train_x.shape)
print("Shape Test Y:", test_x.shape)


# Simple model for sentence / document classification using CNN + global max pooling
model = Sequential()
model.add(Conv1D(filters=250, kernel_size=3, padding='same'))
model.add(GlobalMaxPooling1D())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=10, batch_size=32)


model.summary()

