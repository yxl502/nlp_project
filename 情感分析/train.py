import pandas as pd

neg = pd.read_excel('neg.xls', header=None, index_col=None)
pos = pd.read_excel('pos.xls', header=None, index_col=None)

print(neg.head())
print(pos.head())


pos['mark'] = 1
neg['mark'] = 0

from tensorflow.keras.preprocessing import sequence

import jieba

import pandas as pd

from sklearn.model_selection import train_test_split

cut_word = lambda x: list(jieba.cut(x))

pn_all = pd.concat([pos, neg], ignore_index=True)

print(pn_all.head())

pn_all['words'] = pn_all[0].apply(cut_word)

print(pn_all['words'])

comment = pd.read_excel('sum.xls')
print(comment.head())


comment = comment[comment['rateContent'].notnull()]

comment['words'] = comment['rateContent'].apply(cut_word)

pn_comment = pd.concat([pn_all['words'], comment['words']], ignore_index=True)

print(pn_comment.head())


w = []
for i in pn_comment:
    w.extend(i)

dicts = pd.DataFrame(pd.Series(w).value_counts())
print(len(dicts))
del w, pn_comment

dicts['id'] = list(range(1, len(dicts) + 1))

get_sent = lambda x: list(dicts['id'][x])

pn_all['sent'] = pn_all['words'].apply(get_sent)

max_length = 50
pn_all['sent'] = list(sequence.pad_sequences(pn_all['sent'], maxlen=max_length))

import numpy as np
x_all = np.array(list(pn_all['sent']))
y_all = np.array(list(pn_all['mark']))

x_train, x_test, y_train, y_test = train_test_split(
    x_all, y_all, test_size=0.25
)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_train)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, LSTM

model = Sequential()
model.add(Embedding(len(dicts) + 1, 256, input_length=max_length))
model.add(LSTM(128))

model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

print(model.summary())


if __name__ == '__main__':

    import time

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    timeA = time.time()
    model.fit(x_train, y_train, batch_size=16, epochs=10)

    timeB = time.time()

    print('time cost:', int(timeB - timeA))

    model.save('./model.h5')

