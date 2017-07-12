from keras.layers import Dense, Activation
from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential
import numpy as np
context = 3
hidden_neurons = 50
batch_size = 60
cycles = 5
epochs_per_cycle = 3 #default 1
def create_tesla_text_from_file(textfile="tesla.txt"):
	clean_text_chunks = []
	with open(textfile, 'r', encoding='utf-8') as text:
		for line in text:
			clean_text_chunks.append(line)
	clean_text = (" ".join(clean_text_chunks)).lower()
	text_as_list = clean_text.split()
	return text_as_list
text_as_list =  create_tesla_text_from_file()
distinct_words = set(text_as_list)
number_of_words = len(distinct_words)
word2index = dict((w, i) for i, w in enumerate(distinct_words)) 
index2word = dict((i, w) for i, w in enumerate(distinct_words))
def create_word_indices_for_text(text_as_list):
	input_words = []
	label_word = []
	for i in range(0,len(text_as_list) - context):
		input_words.append((text_as_list[i:i+context]))
		label_word.append((text_as_list[i+context]))
	return input_words, label_word
input_words,label_word = create_word_indices_for_text(text_as_list)
input_vectors = np.zeros((len(input_words), context, number_of_words), dtype=np.int16) 
vectorized_labels = np.zeros((len(input_words), number_of_words), dtype=np.int16)
for i, input_w in enumerate(input_words): 
	for j, w in enumerate(input_w): 
		input_vectors[i, j, word2index[w]] = 1 
		vectorized_labels[i, word2index[label_word[i]]] = 1
model = Sequential()
model.add(SimpleRNN(hidden_neurons, return_sequences=False, input_shape=(context,number_of_words), unroll=True))
model.add(Dense(number_of_words))
model.add(Activation("softmax"))
model.compile(loss="mean_squared_error", optimizer="sgd")
for cycle in range(cycles): 	
	print(">-<" * 50) 
	print(" Cycle: %d" % (cycle+1)) 
	model.fit(input_vectors, vectorized_labels, batch_size = batch_size, epochs = epochs_per_cycle) 	
	test_index = np.random.randint(len(input_words))
	test_words = input_words[test_index]
	print("Generating test from test index %s with words %s: " % (test_index, test_words))
	input_for_test = np.zeros((1, context, number_of_words))
	for i, w in enumerate(test_words): 
		input_for_test[0, i, word2index[w]] = 1 
	predictions_all_matrix = model.predict(input_for_test, verbose = 0)[0] 
	predicted_word = index2word[np.argmax(predictions_all_matrix)] 
	print("THE COMPLETE RESULTING SENTENCE IS: %s %s" % (" ".join(test_words), predicted_word))
	print()#put more cycles in if what you see here is gibberish
