import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from collections import Counter
from tensorflow import keras


def open_file(filename, mode='r'):
    return open(filename, mode, encoding='utf8', errors='ignore')


def read_file(filename):
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(content))
                    labels.append(label)

            except:
                pass

    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    data_train, lab = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.append(content)

    counter = Counter(all_data)

    count_pairs = counter.most_common(vocab_size - 1)

    words, temp = list(zip(*count_pairs))

    words = ['<PAD>'] + list(words)

    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    with open_file(vocab_dir) as fp:
        words = [i.strip() for i in fp.readlines()]

    word_to_id = dict(zip(words, range(len(words))))

    return words, word_to_id


def read_category():
    categories = ['体育', '财经', '房产', '家居', '教育',
                  '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


def to_words(content, words):
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    x_pad = keras.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = keras.utils.to_categorical(label_id, num_classes=len(cat_to_id))
    return x_pad, y_pad


import os

train_dir = 'cnews.train.txt'
test_dir = 'cnews.test.txt'
val_dir = 'cnews.val.txt'
vacab_dir = 'cnews.vocab.txt'

save_path = 'best_validation'

vocab_size = 5000
if not os.path.exists(vacab_dir):
    build_vocab(train_dir, vacab_dir, vocab_size)

categories, cat_to_id = read_category()

words, word_to_id = read_vocab(vacab_dir)

vocab_size = len(words)

seq_length = 600

x_train, y_train = process_file(train_dir, word_to_id, cat_to_id,
                                seq_length)

x_val, y_val = process_file(val_dir, word_to_id, cat_to_id,
                            seq_length)

x_test, y_test = process_file(test_dir, word_to_id, cat_to_id,
                              seq_length)

import tensorflow as tf
from matplotlib.pyplot import MultipleLocator


def TextRNN():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size + 1, 128, input_length=600))
    model.add(tf.keras.layers.LSTM(128))
    model.add(tf.keras.layers.BatchNormalization(epsilon=1e-6, axis=1))
    model.add(tf.keras.layers.Dense(256, activation='relu'))

    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    return model


model = TextRNN()
print(model.summary())

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    with strategy.scope():
        model = TextRNN()
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['categorical_accuracy'])

    history = model.fit(x_train, y_train, batch_size=64, epochs=20,
                        validation_data=(x_val, y_val))

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'SimHei'


    def plot_acc_loss(history):
        plt.subplot(211)
        plt.title('acc')
        plt.plot(range(1, 21), history.history['categorical_accuracy'],
                 linestyle='-', color='g', label='train')

        plt.plot(range(1, 21), history.history['val_categorical_accuracy'],
                 linestyle='-.', color='b', label='test')

        plt.legend(loc='best')

        x_major_locator = MultipleLocator(1)
        ax = plt.gca()

        ax.xaxis.set_major_locator(x_major_locator)

        plt.tick_params(axis='both', which='major', labelsize=7)

        plt.xlabel('epoch')
        plt.ylabel('acc')

        plt.subplot(212)
        plt.title('loss')
        plt.plot(range(1, 21), history.history['loss'], linestyle='-',
                 color='g', label='train')
        plt.plot(range(1, 21), history.history['val_loss'], linestyle='-.',
                 color='b', label='test')
        plt.legend(loc='best')

        x_major_locator = MultipleLocator(1)
        ax = plt.gca()

        ax.xaxis.set_major_locator(x_major_locator)
        plt.tick_params(axis='both', which='major', labelsize=7)

        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.tight_layout()

        plt.savefig('acc-loss.png')
        plt.show()


    plot_acc_loss(history)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model.save(os.path.join(save_path, 'model.h5'))
    del model
