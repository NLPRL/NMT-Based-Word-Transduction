import os
import pickle

path = "/home/saurav/Documents/hindi-wikipedia-articles-55000/"

def load_doc(filename):
	# open the file as read only
	file = pickle.load(open(filename, 'rb'))
	return file


def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


raw_text = ""
for filename in  os.listdir(path)[:1500]:
    if filename.endswith(".pkl"):
        text = load_doc(os.path.join(path, filename))
        raw_text += '' + text

tokens = raw_text.split()
raw_text = ' '.join(tokens)

# organize into sequences of characters
length = 30
sequences = list()
for i in range(length, len(raw_text)):
	# select sequence of tokens
	seq = raw_text[i-length:i+1]
	# store
	sequences.append(seq)
print('Total Sequences: %d' % len(sequences))

out_filename = 'char_sequences.txt'
save_doc(sequences, out_filename)