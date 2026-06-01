import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

sns.set(style='white')

dataset = pd.read_csv('iris.csv')

dataset.columns = [col.strip().replace(' (cm)', '').replace(' ', '_') for col in dataset.columns]

dataset['sepal_length_width_ratio'] = dataset['sepal_length'] / dataset['sepal_width']
dataset['petal_length_width_ratio'] = dataset['petal_length'] / dataset['petal_width']

dataset = dataset[
    [
        'sepal_length',
        'sepal_width',
        'petal_length',
        'petal_width',
        'sepal_length_width_ratio',
        'petal_length_width_ratio',
        'target'
    ]
]

train_data, test_data = train_test_split(
    dataset,
    test_size=0.2,
    random_state=44,
    stratify=dataset['target']
)

X_train = train_data.drop('target', axis=1).values.astype('float32')
y_train = train_data['target'].values.astype('int32')

X_test = test_data.drop('target', axis=1).values.astype('float32')
y_test = test_data['target'].values.astype('int32')

logreg = LogisticRegression(
    C=0.0001,
    solver='lbfgs',
    max_iter=1000
)

logreg.fit(X_train, y_train)

predictions_lr = logreg.predict(X_test)

cm_lr = confusion_matrix(y_test, predictions_lr)

f1_lr = f1_score(y_test, predictions_lr, average='micro')
prec_lr = precision_score(y_test, predictions_lr, average='micro')
recall_lr = recall_score(y_test, predictions_lr, average='micro')

train_acc_lr = accuracy_score(y_train, logreg.predict(X_train)) * 100
test_acc_lr = accuracy_score(y_test, predictions_lr) * 100

rf_clf = RandomForestClassifier(
    n_estimators=100,
    random_state=44
)

rf_clf.fit(X_train, y_train)

predictions_rf = rf_clf.predict(X_test)

cm_rf = confusion_matrix(y_test, predictions_rf)

f1_rf = f1_score(y_test, predictions_rf, average='micro')
prec_rf = precision_score(y_test, predictions_rf, average='micro')
recall_rf = recall_score(y_test, predictions_rf, average='micro')

train_acc_rf = accuracy_score(y_train, rf_clf.predict(X_train)) * 100
test_acc_rf = accuracy_score(y_test, predictions_rf) * 100

def plot_cm(cm, target_name, title="Confusion Matrix", normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    plt.figure(figsize=(10, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(target_name))
    plt.xticks(tick_marks, target_name, rotation=45)
    plt.yticks(tick_marks, target_name)

    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm_display = cm

    thresh = cm_display.max() / 1.5

    for i, j in itertools.product(range(cm_display.shape[0]), range(cm_display.shape[1])):
        plt.text(
            j,
            i,
            "{:.4f}".format(cm_display[i, j]) if normalize else "{:,}".format(cm_display[i, j]),
            horizontalalignment="center",
            color="white" if cm_display[i, j] > thresh else "black"
        )

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel(
        'Predicted Label\naccuracy={:.4f}; misclass={:.4f}'.format(
            accuracy,
            misclass
        )
    )

    plt.savefig('ConfusionMatrix.png', dpi=120)
    plt.show()

target_name = np.array(['setosa', 'versicolor', 'virginica'])

plot_cm(
    cm_lr,
    target_name,
    title='Confusion Matrix (Logistic Regression)',
    normalize=True
)

importances = rf_clf.feature_importances_

labels = dataset.columns[:-1]

feature_df = pd.DataFrame(
    {
        'feature': labels,
        'importance': importances
    }
)

feature_df = feature_df.sort_values(
    by='importance',
    ascending=False
)

plt.figure(figsize=(10, 6))

ax = sns.barplot(
    data=feature_df,
    x='importance',
    y='feature'
)

ax.set_xlabel('Importance')
ax.set_ylabel('Feature')
ax.set_title('Random Forest Feature Importances')

plt.tight_layout()

plt.savefig('FeatureImportance.png', dpi=120)

plt.show()

with open('scores.txt', 'w') as score:

    score.write("RANDOM FOREST CLASSIFIER\n")
    score.write("------------------------\n")
    score.write("Train Accuracy : %.2f%%\n" % train_acc_rf)
    score.write("Test Accuracy  : %.2f%%\n" % test_acc_rf)
    score.write("F1 Score       : %.4f\n" % f1_rf)
    score.write("Recall Score   : %.4f\n" % recall_rf)
    score.write("Precision Score: %.4f\n" % prec_rf)

    score.write("\n")

    score.write("LOGISTIC REGRESSION\n")
    score.write("-------------------\n")
    score.write("Train Accuracy : %.2f%%\n" % train_acc_lr)
    score.write("Test Accuracy  : %.2f%%\n" % test_acc_lr)
    score.write("F1 Score       : %.4f\n" % f1_lr)
    score.write("Recall Score   : %.4f\n" % recall_lr)
    score.write("Precision Score: %.4f\n" % prec_lr)

print("Training completed successfully.")
print("Files generated:")
print("1. ConfusionMatrix.png")
print("2. FeatureImportance.png")
print("3. scores.txt")