from sklearn import metrics
from tensorflow.keras.models import load_model
from train import x_test, y_test

model = load_model('./model.h5')
y_pred = model.predict_classes(x_test)
acc = metrics.accuracy_score(y_test, y_pred)

print(acc)
print(metrics.classification_report(y_test, y_pred))

cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)