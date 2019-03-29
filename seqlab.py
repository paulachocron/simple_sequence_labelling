import pandas as pd
import json
import numpy as np
import argparse

from gensim.models import KeyedVectors

from keras.models import Model, Input, load_model
from keras.layers import Dense, Activation, Embedding, LSTM, TimeDistributed, Dropout, Bidirectional    

from keras.preprocessing.sequence import pad_sequences        
from keras.utils import to_categorical


# from keras_contrib.layers import CRF

def preprocess_text(sentences):
    """Clean accent marks and make lowercase (social media-friendly).
    Assumes sentences is tokenized.
    """
    for sentence in sentences:
        for word in sentence:
            word = word.replace('á', 'a')
            word = word.replace('é', 'e')
            word = word.replace('í', 'i')
            word = word.replace('ó', 'o')
            word = word.replace('ú', 'u')
            word = word.replace('ü', 'u')
            word = word.lower()


def get_data(file):
    """Get data from a file with a column of words and a colum of tags."""

    sentences, labels = [],[]
    with open(file, encoding="UTF-8") as f:
        sentence,sent_labels = [],[]
        for line in f:
            line = line.rstrip()
            if len(line) == 0:
                sentences.append(sentence)
                labels.append(sent_labels)
                sentence,sent_labels = [],[]
            else:
                word, label = line.split()
                sentence.append(word)
                sent_labels.append(label)
    return sentences, labels


def process_sequences(sequences,formatter):
    """Receives sequences of text tokens and a dictionary text->number.
    Returns the sequences converted in numbers and padded.
    """
    max_l = max([len(t) for t in sequences])
    extra = [len(formatter.keys())+1 for i in range(max_l)]
    num_sequences = [[formatter[w] for w in s if w in formatter] or extra for s in sequences]
    sequences = pad_sequences(maxlen=max_l, sequences=num_sequences, padding="post", value=formatter['--PAD--'])
    return sequences


def train_eval(data_path, model_name, option='simple',emb_path=None):
    """Train a model with the data in path.
    Save it (and the formatting) as model_name.
    If option is 'emb', emb_path is the path to the embedding to be used.
    """
    
    # get the data
    try:
        X_train, y_train = get_data(data_path+'/train')
        X_val, y_val = get_data(data_path+'/val')
        X_test, y_test = get_data(data_path+'/test')
    except:
        raise Exception("Some data file does not exist")
    
    # preprocess the texts
    for X in [X_train, X_val, X_test]:
        preprocess_text(X)


    # Keras needs the sequences to be numerical and padded, as well as the labels
    # We will need all the words and labels for this

    words = list(set([w for sent in X_train+X_val+X_test for w in sent]))
    labels = list(set([l for sent in y_train for l in sent]))

    words.append('--PAD--')
    # labels.append('--PAD--')

    n_labels = len(labels)
    n_words = len(words)

    words2num = {word: i for i, word in enumerate(words)}
    labels2num = {label: i for i, label in enumerate(labels)}
   
    # a trick for NER...
    if 'O' in labels2num:
        labels2num['--PAD--']=labels2num['O']
    else:
        labels2num['--PAD--']=enumerate(labels)+1

    [X_train_num,X_val_num,X_test_num] = [process_sequences(X,words2num) for X in [X_train,X_val,X_test]]

    [y_train_num,y_val_num,y_test_num] = [process_sequences(y,labels2num) for y in [y_train,y_val,y_test]]
    [y_train_num,y_val_num,y_test_num] = [[to_categorical(i, num_classes=n_labels) for i in y] for y in [y_train_num,y_val_num,y_test_num]]


    if option=='emb':
        try:
            emb_dict = KeyedVectors.load(emb_path)

        except:
            raise Exception("Embedding file does not exist")

        emb_matrix = np.zeros((len(words), emb_dict.vector_size))


        for i, w in enumerate(words):
            # Build a matrix for the indexes with the vector values of corresponding words
            # If the word does not exist in the embedding, keep zeros
            if w in emb_dict:
                emb_matrix[i] = emb_dict[w]

    # We build a Bidirectional LSTM
    input = Input(shape=(None,))
    if option=='emb':
        model = Embedding(input_dim=n_words, output_dim=emb_dict.vector_size, weights=[emb_matrix])(input)
    else:
        model = Embedding(input_dim=n_words, output_dim=50)(input)
    model = Dropout(0.1)(model)
    model = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(model)
    out = TimeDistributed(Dense(n_labels, activation="softmax"))(model)  # TimeDistributed keeps the outputs for each sequence separated
    # crf = CRF(n_labels)  # CRF layer
    # out = crf(model)  
    model = Model(input, out)
 
    if option=='crf':
        crf = CRF(n_labels)  # CRF layer
        out = crf(model) 
        model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

    else:
        model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

    # Fit the model using the validation data
    model.fit(X_train_num
            , np.array(y_train_num)
            , batch_size=32
            , epochs = 5
            , validation_data=(X_val_num,np.array(y_val_num))
            , verbose=1)

    # Save the model
    model.save('{}.hdf5'.format(model_name), overwrite=True)
    formatter = {'labels': labels, 'words': 'words','words2num':words2num, 'labels2num':labels2num}
    with open('{}-preproc.json'.format(model_name), 'w+') as f:
        json.dump(formatter, f)

    # Evaluate the model on the test data
    predictions = model.predict(X_test_num)
    results = model.evaluate(X_test_num, np.array(y_test_num))

    print("Overall results for the predictions: {}".format(results))

    # This values are not very clear because of class imbalance
    # Make a better evaluation
    predictions = np.argmax(predictions, axis=-1)
    predictions = [[labels[i] for i in pred] for pred in predictions]
    evaluate(y_test,predictions,labels)

    return(predictions) 



def test_eval(path, model_name=None):
    
    X_test, y_test = get_data(path+'/test')
    
    model, preprocessor = get_model(model_name)

    preprocess_text(X_test)
    X_test_num = process_sequences(X_test, preprocessor['words2num'])

    predictions = model.predict(X_test_num)

    predictions = np.argmax(predictions, axis=-1)
    predictions = [[preprocessor['labels'][i] for i in pred] for pred in predictions]

    evaluate(y_test,predictions,preprocessor['labels'])

    return(predictions) 



def evaluate(y_test,preds,labels):
    """ Evaluate predictions by computing precision and recall of each class"""

    # get all pairs of real value - predictions
    # all_preds = zip([l for sentence in y_test for l in sentence],[p for sentence in preds for p in sentence])
    all_preds = []

    for i in range(len(y_test)):
        for j in range(len(y_test[i])):
            all_preds.append((y_test[i][j], preds[i][j]))

    # print(preds[:200])
    for l in labels:
        print("precision {}".format(l))
        print(float(len([w for w in all_preds if w[0]==l and w[1]==l])/len([w for w in all_preds if w[1]==l])))
        print("recall {}".format(l))
        print(float(len([w for w in all_preds if w[0]==l and w[1]==l])/len([w for w in all_preds if w[0]==l])))


def get_model(model_name):
    """Get saved model and preprocessor."""
    try:
        model_file = '{}.hdf5'.format(model_name)
        prepr_file = '{}-preproc.json'.format(model_name)
    except:
        raise Exception('Model not found')

    model = load_model(model_file)

    with open(prepr_file) as json_data:
        preprocessor = json.load(json_data)

    return model, preprocessor



def tag(sentences,model_name,dest_file):
    """Tag sentences using model from model_name.
    Assumes sentences are tokenized. 
    Save values to dest_file.
    """

    model, preprocessor = get_model(model_name)

    preprocess_text(sentences)

    X = process_sequences(sentences, preprocessor['words2num'])

    predictions = model.predict(X)

    predictions = np.argmax(predictions, axis=-1)
    predictions = [[preprocessor['labels'][i] for i in pred] for pred in predictions]

    with open(dest_file, 'w+') as f:
        for s,preds in zip(sentences, predictions):
            for w, l in zip(s, preds):
                f.write('{} {}\n'.format(w, l))            
            f.write('\n')

# test_eval('data/spanish_clean', model_name='ner-es-emb')
# train_eval('data/spanish_clean', model_name='ner-es', option='simple')

# tag(["donald trump pregunto que hacer a sus ministros con respecto a africa".split()],'ner-es-emb', 'test.txt')


def main():
    parser = argparse.ArgumentParser(description="Sequence Labelling demo")

    parser.add_argument("--action", default='eval', help="one from [tag, eval, train]")
    parser.add_argument("--model", default='emb', help="one from [simple,emb]")

    args = parser.parse_args()

    action = args.action
    if action not in ('tag', 'eval', 'train'):
        raise NameError('Supported actions are: [tag, eval, train]')

    model = args.model
    if model not in ['simple','emb']:
        raise NameError('Supported models are simple or emb')


    if action=='eval':
        if model=='emb':
            test_eval('data/spanish_clean', model_name='models/taggers/ner-es-emb')
        else:
            test_eval('data/spanish_clean', model_name='models/taggers/ner-es')

    if action=='train':
        if model=='emb':
            train_eval('data/spanish_clean', model_name='models/taggers/ner-es-emb', option='emb', emb_path='models/vectors/spanish.vec')
        else:
            test_eval('data/spanish_clean', model_name='models/taggers/ner-es',option='simple')

    else:
        sample_sentences = ["donald trump pregunto que hacer a sus ministros con respecto a africa".split(),
                            "argentina es un pais lleno de oportunidades".split()]
        if model=='emb':
            tag(sample_sentences,'models/taggers/ner-es-emb', 'test.txt')
        else:
            tag(sample_sentences,'models/taggers/ner-es', 'test.txt')



if __name__ == "__main__":
    main()
