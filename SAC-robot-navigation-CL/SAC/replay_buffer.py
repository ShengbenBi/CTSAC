import random
from collections import deque   # deque stands for "double-ended queue"
from itertools import islice

import numpy as np  # numpy is an open-source Python library for scientific computing, used for fast processing of n-dimensional arrays. Imported as np.

class ReplayBuffer(object):  # Define the ReplayBuffer class
    def __init__(self, buffer_size, random_seed=1):
        """
        The right side of the deque contains the most recent experiences
        # The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size      # Represents the maximum size of the buffer
        self.count = 0                      # Represents the number of elements currently in the buffer
        self.buffer = deque()               # A double-ended queue
        random.seed(random_seed)            # The seed is the starting point for the random number generator. Setting a seed ensures reproducibility: using the same seed produces the same sequence of random numbers.

    def add(self, s, a, r, s_, d, end):  # self refers to the instance of the class
        # d indicates completion, end indicates termination (including completion or collision)
        experience = (s, a, r, s_, d, end)
        if self.count < self.buffer_size:  # If the buffer is not full
            self.buffer.append(experience)  # Append the experience to the right end of the deque
            self.count += 1
        else:                              # If the buffer is full
            self.buffer.popleft()          # Remove the element from the left end of the deque
            self.buffer.append(experience) # Append the experience to the right end of the deque

    def size(self):
        return self.count                  # Return the number of elements in the buffer

    def sample_batch(self, batch_size, keys_num):
        if self.count < batch_size * keys_num / 2:  # If the number of elements in buffer is less than batch_size * keys_num / 2
            batch_size = self.count // keys_num

        s_batch, a_batch, r_batch, s2_batch, d_batch = [], [], [], [], []

        for _ in range(batch_size):
            while True:
                end_index = random.randint(keys_num, self.count)
                sequence = list(islice(self.buffer, end_index - keys_num, end_index))
                # sequence = self.buffer[end_index - keys_num : end_index]
                s_seq, a_seq, r_seq, s2_seq, d_seq, end_seq = zip(*sequence)

                # Check that the 'end' flag of the first keys_num - 1 entries is not equal to 1
                if all(end != 1 for end in end_seq[:-1]):
                    s_batch.append(s_seq)
                    a_batch.append(a_seq)
                    r_batch.append(r_seq)
                    s2_batch.append(s2_seq)
                    d_batch.append(d_seq)
                    break  # Break out of the while loop to process the next batch

        s_batch = np.array(s_batch)
        a_batch = np.array(a_batch).reshape(batch_size, keys_num, -1)
        r_batch = np.array(r_batch).reshape(batch_size, keys_num, -1)
        s2_batch = np.array(s2_batch)
        d_batch = np.array(d_batch).reshape(batch_size, keys_num, -1)

        return s_batch, a_batch, r_batch, s2_batch, d_batch

    def clear(self):
        self.buffer.clear()  # Clear the buffer
        self.count = 0       # Reset the buffer element count to zero
