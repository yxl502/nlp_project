import numpy as np
import seaborn as sns
from 文本分类.lstm_text import x_test, y_test

from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'SimHei'

# from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

model = load_model('./best_validation/model.h5')

y_pre = model.predict(x_test)

confm = confusion_matrix(np.argmax(y_pre, axis=1),
                         np.argmax(y_test, axis=1))

print(classification_report(np.argmax(y_pre, axis=1),
                            np.argmax(y_test, axis=1)))

plt.figure(figsize=(8, 8))
sns.heatmap(confm.T, square=True, annot=True,
            fmt='d', cbar=False, linewidths=.8,
            cmap='YlGnBu')

plt.xlabel('true:', size=14)
plt.ylabel('pre:', size=14)

categories = ['体育', '财经', '房产', '家居', '教育',
                  '科技', '时尚', '时政', '游戏', '娱乐']
plt.xticks(np.arange(10) + 0.5, categories, size=12)
plt.yticks(np.arange(10) + 0.3, categories, size=12)
plt.savefig('confuse.png')
plt.show()