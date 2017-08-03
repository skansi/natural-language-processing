
from keras.models import Sequential
from keras.layers.core import Dense#,
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
text_as_list=["who","are","you","that","you","do","not","know","your","history"]
embedding_size = 300
context = 2
distinct_words = set(text_as_list)
number_of_words = len(distinct_words)
word2index = dict((w, i) for i, w in enumerate(distinct_words)) 
index2word = dict((i, w) for i, w in enumerate(distinct_words))
def create_word_context_and_main_words_lists(text_as_list):
	input_words = []
	label_word = []
	for i in range(0,len(text_as_list)):
		label_word.append((text_as_list[i]))
		context_list = []
		if i >= context and i<(len(text_as_list)-context):
			context_list.append(text_as_list[i-context:i])
			context_list.append(text_as_list[i+1:i+1+context])
			context_list = [x for subl in context_list for x in subl]
		elif i<context:
			context_list.append(text_as_list[:i])
			context_list.append(text_as_list[i+1:i+1+context])
			context_list = [x for subl in context_list for x in subl]
		elif i>=(len(text_as_list)-context):
			context_list.append(text_as_list[i-context:i])
			context_list.append(text_as_list[i+1:])
			context_list = [x for subl in context_list for x in subl]
		input_words.append((context_list))
	return input_words, label_word
input_words,label_word = create_word_context_and_main_words_lists(text_as_list)
input_vectors = np.zeros((len(text_as_list), number_of_words), dtype=np.int16) 
vectorized_labels = np.zeros((len(text_as_list), number_of_words), dtype=np.int16)
for i, input_w in enumerate(input_words): 
	for j, w in enumerate(input_w): 
		input_vectors[i, word2index[w]] = 1 
		vectorized_labels[i, word2index[label_word[i]]] = 1
word2vec = Sequential()
word2vec.add(Dense(embedding_size, input_shape=(number_of_words,), activation="linear", use_bias=False))
word2vec.add(Dense(number_of_words, activation="softmax", use_bias=False))
word2vec.compile(loss="mean_squared_error", optimizer="sgd", metrics=['accuracy'])
word2vec.fit(input_vectors, vectorized_labels, epochs=1500, batch_size=10, verbose=1)
metrics = word2vec.evaluate(input_vectors, vectorized_labels, verbose=1)
print("%s: %.2f%%" % (word2vec.metrics_names[1], metrics[1]*100))
embedding_weight_matrix = word2vec.get_weights()[0]
word2vec.save_weights("all_weights.h5")
pca = PCA(n_components=2)
pca.fit(embedding_weight_matrix)
results = pca.transform(embedding_weight_matrix) 
x = np.transpose(results).tolist()[0]
y = np.transpose(results).tolist()[1]
n = list(word2index.keys()) 
fig, ax = plt.subplots()
ax.scatter(x, y)
for i, txt in enumerate( n ):
    ax.annotate( txt, ( x[i], y[i]) )
plt.savefig('word_vectors_in_2D_space.png')
plt.show()