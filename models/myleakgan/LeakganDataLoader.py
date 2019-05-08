import numpy as np


class DataLoader():
    def __init__(self, batch_size, seq_length, end_token=0):
        self.batch_size = batch_size
        self.token_stream = []
        self.seq_length = seq_length
        self.end_token = end_token

    def create_batches(self, data_file):
        self.token_stream = []

        with open(data_file, 'r') as raw:
            for line in raw:
                line = line.strip().split()
                parse_line = [int(x) for x in line]
                if len(parse_line) > self.seq_length:
                    self.token_stream.append(parse_line[:self.seq_length])
                else:
                    while len(parse_line) < self.seq_length:
                        parse_line.append(self.end_token)
                    if len(parse_line) == self.seq_length:
                        self.token_stream.append(parse_line)

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0


class DisDataloader():
    def __init__(self, batch_size, seq_length):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])
        self.seq_length = seq_length

    def load_train_data(self, positive_file, negative_file):
        # Load data
        positive_examples = []
        negative_examples = []
        with open(positive_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == self.seq_length:
                    positive_examples.append(parse_line)
        with open(negative_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == self.seq_length:
                    negative_examples.append(parse_line)
        self.sentences = np.array(positive_examples + negative_examples)

        self.sentences = split_sentence(self.sentences)
        # Generate labels
        positive_labels = [[0, 1]] * len(positive_examples)
        negative_labels = [[1, 0]] * len(negative_examples)
        self.labels = np.concatenate([positive_labels, negative_labels] * self.seq_length)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)

        self.pointer = 0

    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0


def split_sentence(input_data):
    """
    input_data: numpy.array with [batch_size x seq_length]
    """
    # make sure this is 2d array
    assert input_data.ndim == 2
    seq_length = input_data.shape[1]
    padding_value = input_data[-1,-1]
    # Load data 
    datasets = []
    for i in range(1, seq_length + 1):
        padding_shape = [[0,0], [0, seq_length - i]]
        data = np.pad(input_data[:,:i], padding_shape, 'constant', constant_values=padding_value)
        datasets.append(data)
    return np.concatenate(datasets)