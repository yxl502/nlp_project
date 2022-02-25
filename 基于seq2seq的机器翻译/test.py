import numpy as np
import tensorflow as tf
from train import *

# checkpoint_path = './training_checkpoints/ckpt-25'

# checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder,
#                                  decoder=decoder)
# 从文件恢复模型参数
checkpoint.restore(tf.train.latest_checkpoint('./training_checkpoints'))

def evaluate(sentence):
    '''
    sentence：需要翻译的句子
    '''
    attention_plot = np.zeros((max_length_targ, max_length_inp))
    sentence = preprocess_sentence(sentence)
    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    result = ''
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)
    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()
        result += targ_lang.index_word[predicted_id] + ' '
        if targ_lang.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot
        # 预测的ID被输送回模型
        dec_input = tf.expand_dims([predicted_id], 0)
    return result, sentence, attention_plot

# 执行翻译
def translate(sentence):
    '''
    sentence：要翻译的句子
    '''
    result, sentence, attention_plot = evaluate(sentence)
    print('输入：%s' % (sentence))
    print('翻译结果：{}'.format(result))

print(translate('我生病了。'))
print(translate('为什么不？'))
print(translate('让我一个人呆会儿。'))
print(translate('打电话回家！'))
print(translate('我了解你。'))

print(translate('你确定？'))


