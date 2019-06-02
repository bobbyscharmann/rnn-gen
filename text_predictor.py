"""
Install all Python modules from requirements.txt (highly recommend using a
Python virtual environment)

pip install -r requirements.txt

To run use:

    python text_predictor.py sherlock

Since it can be confusing, a note on epochs, iterations, and batch size:
    Say you have a dataset of 10 examples (or samples). You have a batch size 
    of 2, and you've specified you want the algorithm to run for 3 epochs.

    Therefore, in each epoch, you have 5 batches (10/2 = 5). Each batch gets 
    passed through the algorithm, therefore you have 5 iterations per epoch. 
    Since you've specified 3 epochs, you have a total of 15 iterations (
    5*3 = 15) for training.
"""
import tensorflow as tf
from data_provider import DataProvider
from rnn_model import RNNModel
import sys
import matplotlib
import numpy as np
import time
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Args - exit early if an incorrect number is provided by the user
# TODO: Check if the user passed in a valid dataset
if len(sys.argv) != 2:
    print("Please select a dataset.")
    print("Usage: python text_predictor.py <dataset>")
    print("Available datasets: kanye, shakespeare, wikipedia, reuters, " +
          "hackernews, war_and_peace, sherlock")
    exit(1)
else:
    dataset = sys.argv[1]

# I/O
data_dir = "./data/" + dataset
tensorboard_dir = data_dir + "/tensorboard/" + str(time.strftime("%Y-%m-%d_%H-%M-%S"))
input_file = data_dir + "/input.txt"
output_file = data_dir + "/output.txt"
output = open(output_file, "w")
output.close()

# Hyperparameters
# Batch size pertains to the amount of training samples to consider at a time 
# for updating your network weights. So, in a feedforward network, let's say 
# you want to update your network weights based on computing your gradients 
# from one word at a time, your batch_size = 1. As the gradients are computed 
# from a single sample, this is computationally very cheap. On the other hand,
# it is also very erratic training. The higher the batch size, the more memory 
# space you'll need.
BATCH_SIZE = 32
SEQUENCE_LENGTH = 25
LEARNING_RATE = 0.01
DECAY_RATE = 0.97
HIDDEN_LAYER_SIZE = 256
CELLS_SIZE = 2

# Length of the text to sample from the model
TEXT_SAMPLE_LENGTH = 500

# Frequency at which we stop and evaluate the performance of the model
SAMPLING_FREQUENCY = 1000

# Frequency at which we log to the screen and update the plot general status
#information about model performance
LOGGING_FREQUENCY = 1000


def rnn():
    data_provider = DataProvider(data_dir, BATCH_SIZE, SEQUENCE_LENGTH)
    model = RNNModel(data_provider.vocabulary_size, 
                     batch_size=BATCH_SIZE, 
                     sequence_length=SEQUENCE_LENGTH, 
                     hidden_layer_size=HIDDEN_LAYER_SIZE, 
                     cells_size=CELLS_SIZE)

    with tf.Session() as sess:

        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(tensorboard_dir)
        writer.add_graph(sess.graph)
        sess.run(tf.global_variables_initializer())

        # Forward pass and one backward pass of all the training examples
        epoch = 0
        temp_losses = []
        smooth_losses = []

        while True:
            sess.run(tf.assign(model.learning_rate, LEARNING_RATE * (DECAY_RATE ** epoch)))
            data_provider.reset_batch_pointer()
            state = sess.run(model.initial_state)
            for batch in range(data_provider.batches_size):
                inputs, targets = data_provider.next_batch()
                feed = {model.input_data: inputs, model.targets: targets}
                for index, (c, h) in enumerate(model.initial_state):
                    feed[c] = state[index].c
                    feed[h] = state[index].h

                # Iteration is the number of times batch data has passed
                # through the neural network - both forward and backwards
                # propagation
                iteration = epoch * data_provider.batches_size + batch
                summary, loss, state, _ = sess.run([summaries, model.cost, model.final_state, model.train_op], feed)
                writer.add_summary(summary, iteration)
                temp_losses.append(loss)

                if iteration % SAMPLING_FREQUENCY == 0:
                    sample_model(sess, data_provider, iteration)

                if iteration % LOGGING_FREQUENCY == 0:
                    smooth_loss = np.mean(temp_losses)
                    smooth_losses.append(smooth_loss)
                    temp_losses = []
                    plot(smooth_losses, "iterations (thousands)", "loss")
                    print('{{"metric": "iteration", "value": {}}}'.format(iteration))
                    print('{{"metric": "epoch", "value": {}}}'.format(epoch))
                    print('{{"metric": "loss", "value": {}}}'.format(smooth_loss))
            epoch += 1

def sample_model(sess, data_provider, iteration):
    model = RNNModel(data_provider.vocabulary_size, 
                     batch_size=1, 
                     sequence_length=1, 
                     hidden_layer_size=HIDDEN_LAYER_SIZE, 
                     cells_size=CELLS_SIZE, 
                     training=False)
    text = model.sample(sess, 
                        data_provider.chars, 
                        data_provider.vocabulary, 
                        TEXT_SAMPLE_LENGTH)
    
    output = open(output_file, "a")
    output.write("Iteration: " + str(iteration) + "\n")
    output.write(str(text))
    output.write("\n")
    output.close()

def plot(data, x_label, y_label):
    # Plot the data - assumes an array with the X-axis reflecting the index
    plt.plot(range(len(data)), data)

    # Title with the dataset name
    plt.title(dataset)

    # Give the axis labels
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Save the figure for post analysis
    plt.savefig(data_dir + "/" + y_label + ".png", bbox_inches="tight")
    plt.close()


if __name__ == '__main__':
    # Print out some basic debug information 
    print("Selected dataset: " + str(dataset))
    print("Batch size: " + str(BATCH_SIZE))
    print("Sequence length: " + str(SEQUENCE_LENGTH))
    print("Learning rate: " + str(LEARNING_RATE))
    print("Decay rate: " + str(DECAY_RATE))
    print("Hidden layer size: " + str(HIDDEN_LAYER_SIZE))
    print("Cells size: " + str(CELLS_SIZE))
    rnn()
