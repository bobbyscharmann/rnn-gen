"""
A DataProvider is a class which provides some predefined data types from some
source (TFRecord, etc). The most basic function of a
data provider is the `Get` operation where one requests one or more types of
data, or 'items':
provider.get(items=['image', 'sentence', 'class'])
More concretely, a data provider (a subclass of BaseDataProvider) returns a
single tensor for each requested item (data type):

provider = MyDataProvider(...)
image, sentence, clazz = provider.get(['image', 'sentence', 'class'])

In this example, the provider `MyDataProvider` must know how to load each item.
A data provider may be written in a way that the logic necessary to map from
each item to tensor is completely encapsulated within the data_provider itself.
"""
import codecs
import os
import collections
import numpy as np


class DataProvider:
    """Maps a list of requested data items to tensors from a data source.
       All data providers must inherit from DataProvider and implement the Get
       method which returns arbitrary types of data. No assumption is made about the
       source of the data nor the mechanism for providing it.
    """
    def __init__(self, data_dir, batch_size, sequence_length):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        with codecs.open(os.path.join(data_dir, "input.txt"), "r", encoding="utf-8") as file:
            data = file.readlines()
        count_pairs = sorted(collections.Counter(data).items(), key=lambda x: -x[1])
        self.pointer = 0
        self.chars, _ = zip(*count_pairs)
        self.vocabulary_size = len(self.chars)
        self.vocabulary = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.array(list(map(self.vocabulary.get, data)))
        self.batches_size = int(self.tensor.size / (self.batch_size * self.sequence_length))
        if self.batches_size == 0:
            assert False, "Unable to generate batches. Reduce batch_size or sequence_length."

        self.tensor = self.tensor[:self.batches_size * self.batch_size * self.sequence_length]
        inputs = self.tensor
        targets = np.copy(self.tensor)

        targets[:-1] = inputs[1:]
        targets[-1] = inputs[0]
        self.input_batches = np.split(inputs.reshape(self.batch_size, -1), self.batches_size, 1)
        self.target_batches = np.split(targets.reshape(self.batch_size, -1), self.batches_size, 1)
        print("Tensor size: " + str(self.tensor.size))
        print("Batch size: " + str(self.batch_size))
        print("Sequence length: " + str(self.sequence_length))
        print("Batches size: " + str(self.batches_size))
        print("")

    def next_batch(self):
        inputs = self.input_batches[self.pointer]
        targets = self.target_batches[self.pointer]
        self.pointer += 1
        return inputs, targets

    def reset_batch_pointer(self):
        self.pointer = 0
