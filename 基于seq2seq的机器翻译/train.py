import re
import io
import tensorflow as tf
from sklearn.model_selection import train_test_split


def preprocess_sentence(w):
    w = re.sub(r'([?.!,])', r' \1 ', w)
    w = re.sub(r"[' ']+", ' ', w)
    w = '<start> ' + w + ' <end>'
    return w


en_sentence = 'I like this book'
sp_sentence = '我喜欢这本书'
print('预处理前的输出为：', '\n', preprocess_sentence(en_sentence))
print('预处理前的输出为：', '\n', str(preprocess_sentence(sp_sentence)), '\n')


def create_dataset(path, num_examples):
    lines = io.open(path, encoding='utf8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]
                  for l in lines[:num_examples]]
    return zip(*word_pairs)


path_to_file = 'en-ch.txt'
en, sp = create_dataset(path_to_file, None)
print(en[:10])
print(sp[:10])


def max_length(tensor):
    return max(len(t) for t in tensor)


def tokenizer(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(
        tensor, padding='post'
    )

    return tensor, lang_tokenizer


def load_dataset(path, num_example=None):
    targ_lang, inp_lang = create_dataset(path, num_example)
    input_tensor, inp_lang_tokenizer = tokenizer(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenizer(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


num_examples = 2000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(
    path_to_file, num_examples
)

max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)

input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = \
    train_test_split(input_tensor, target_tensor, test_size=0.2)

print(input_tensor_train.shape)
print(input_tensor_val.shape)
print(target_tensor_train.shape)
print(target_tensor_val.shape)


def convert(lang, tensor):
    for t in tensor:
        if t != 0:
            print("%d ----> %s" % (t, lang.index_word[t]))


convert(inp_lang, input_tensor_train[0])
convert(targ_lang, target_tensor_train[0])

buffer_size = len(input_tensor_train)
batch_size = 64
steps_per_epoch = len(input_tensor_train) // batch_size

embedding_dim = 256
units = 1024

vacab_inp_size = len(inp_lang.word_index) + 1
vocab_tar_size = len(targ_lang.word_index) + 1

dataset = tf.data.Dataset.from_tensor_slices(
    (
        input_tensor_train, target_tensor_train
    )
).shuffle(buffer_size)

dataset = dataset.batch(batch_size, drop_remainder=True)
example_input_batch, example_target_batch = next(iter(dataset))


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


encoder = Encoder(vacab_inp_size, embedding_dim, units, batch_size)
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)

print('(batch_size, sequence_length, units) {}'.format(sample_output.shape))
print('(batch_size, units) {}'.format(sample_hidden.shape))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)
        ))

        attention_weights = tf.nn.softmax(
            score, axis=1
        )

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(
    sample_hidden, sample_output
)

print('(batch_size, units) {}'.format(attention_result.shape))
print('(batch_size, sequence_length, 1) {}'.format(attention_weights.shape))

#
# class Decoder(tf.keras.models):
#     def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
#         super(Decoder, self).__init__()
#         self.batch_sz = batch_sz
#         self.dec_units = dec_units
#         self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
#         self.gru = tf.keras.layers.GRU(
#             self.dec_units, return_sequences=True,
#             return_state=True, recurrent_initializer='glorot_uniform'
#         )
#
#         self.fc = tf.keras.layers.Dense(vocab_size)
#         self.attention = BahdanauAttention(self.dec_units)
#
#     def call(self, x, hidden, enc_output):
#         context_vector, attention_weights = self.attention(hidden, enc_output)
#
#         x = self.embedding(x)
#
#         x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
#         output, state = self.gru(x)
#
#         output = tf.reshape(output, (-1, output.shape[2]))
#         x = self.fc(output)
#
#         return x, state, attention_weights
#
#
# decoder = Decoder(vocab_tar_size, embedding_dim, units, batch_size)
# sample_decoder_output, states, attention_weight = decoder(
#     tf.random.uniform((64, 1)), sample_hidden, sample_output
# )
#
# print('(batch_size, vocab_size) {}'.format(sample_decoder_output.shape))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz  # 每次训练所选取的样本数
        self.dec_units = dec_units  # 神经元数量
        # 输入层
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        # 调用注意力模型
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # 编码器输出（enc_output）的形状 == （批大小，最大长度，隐藏层大小）
        context_vector, attention_weights = self.attention(hidden, enc_output)
        # x在通过嵌入层后的形状 == （批大小，1，嵌入维度）
        x = self.embedding(x)
        # x在拼接（concatenation）后的形状 == （批大小，1，嵌入维度 + 隐藏层大小）
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # 将合并后的向量传送到GRU
        output, state = self.gru(x)
        # 输出的形状 == （批大小 * 1，隐藏层大小）
        output = tf.reshape(output, (-1, output.shape[2]))
        # 输出的形状 == （批大小，vocab）
        x = self.fc(output)
        return x, state, attention_weights


# 构建解码器网络结构
decoder = Decoder(vocab_tar_size, embedding_dim, units, batch_size)
sample_decoder_output, states, attention_weight = decoder(tf.random.uniform((64, 1)), sample_hidden, sample_output)
print('解码器输出形状：', '\n', ' (batch_size, vocab size) {}'.format(sample_decoder_output.shape))


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


import os
import time

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder,
                                 decoder=decoder)

def train(inp, targ, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * batch_size, 1)

        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, dec_predictions = decoder(
                dec_input, dec_hidden, enc_output
            )
            loss += loss_function(targ[:, t], predictions)

            dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss


if __name__ == '__main__':

    epoch = 50

    loss = []
    for epoch in range(epoch):
        start = time.time()
        enc_hidden = encoder.initialize_hidden_state()  # 初始化隐藏层
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train(inp, targ, enc_hidden)
            total_loss += batch_loss
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
                loss.append(round(batch_loss.numpy(), 3))
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

    # 损失趋势可视化
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 对字符进行显示设置
    plt.plot(list(range(1, 51)), loss)  # 将损失值绘制成折线图
    plt.title('损失趋势图', fontsize=16)  # 设置折线图标题为损失趋势图
    plt.xlabel('迭代次数')  # 将x轴标签设置为迭代次数
    plt.ylabel('损失值')  # 将y轴标签设置为损失值
    plt.savefig('loss.png')
    plt.show()  # 将图形进行展示
