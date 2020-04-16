import numpy as np
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import ComplementNB
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

# Reading data from folder
training_samples = np.genfromtxt('ml-2020-unibuc-3/train_samples.txt', dtype=None, comments=None, encoding="utf-8", delimiter='\t', names=('id', 'text'))
training_labels_samples = np.genfromtxt('ml-2020-unibuc-3/train_labels.txt', dtype=None, comments=None, names=('id', 'label'))
validation_samples = np.genfromtxt('ml-2020-unibuc-3/validation_samples.txt', dtype=None, comments=None, encoding="utf-8", delimiter='\t', names=('id', 'text'))
validation_labels_samples = np.genfromtxt('ml-2020-unibuc-3/validation_labels.txt', dtype=None, comments=None, names=('id', 'label'))
testing_samples = np.genfromtxt('ml-2020-unibuc-3/test_samples.txt', dtype=None, comments=None, encoding="utf-8", delimiter='\t', names=('id', 'text'))

# What kind of data was extracted from files
training_words = training_samples['text']
training_labels = training_labels_samples['label']
validation_words = validation_samples['text']
validation_labels = validation_labels_samples['label']
testing_data = testing_samples['text']

# After the model is tested on the normal data, it is trained on validation
#training_words = np.append(training_words, validation_words)
#training_labels = np.append(training_labels, validation_labels)

# Transform corpus to a vector of term/token counts
vector = CountVectorizer(lowercase=False, ngram_range=(4, 5), analyzer='char_wb')
training_vector = vector.fit_transform(training_words, training_labels)
validation_vector = vector.transform(validation_words)
testing_vector = vector.transform(testing_data)

# Training tool that allows usage of better training data
scaler = TfidfTransformer()
normalized_training_vector = scaler.fit_transform(training_vector)
normalized_validation_vector = scaler.transform(validation_vector)
normalized_testing_vector = scaler.transform(testing_vector)

# Mathematical representation of a real-world process
model = ComplementNB(alpha=0.005)
model.fit(normalized_training_vector, training_labels)
# Getting predictions
test_prediction = model.predict(normalized_testing_vector)
validation_prediction = model.predict(normalized_validation_vector)

# Report with precision, recall, f1-score and support
print(classification_report(validation_labels, validation_prediction, digits=7))

# Print results in a .csv file
submission = np.vstack((testing_samples['id'], test_prediction)).T
np.savetxt('MOROCOdialect.csv', submission, fmt='%s', delimiter=',', header='id,label', comments='')

# Plot confusion matrix
titles_options = [('Confusion matrix, without normalization', None, 'd', ''), ('Normalized confusion matrix', 'true', 'f', '_norm')]
np.set_printoptions(precision=2)
for title, norm, fmt, extension in titles_options:
    display = plot_confusion_matrix(model, normalized_validation_vector, validation_labels, normalize=norm, cmap=plt.cm.Blues, values_format=fmt)
    display.ax_.set_title(title)
    print(title)
    print(display.confusion_matrix)
    plt.savefig('confusion_matrix' + extension)
    plt.close()