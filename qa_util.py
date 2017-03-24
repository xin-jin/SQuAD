import numpy as np
import os
import qa_data
import copy


def get_minibatches(data, minibatch_size, shuffle=True):
    """
    Iterates through the provided data one minibatch at at time. You can use this function to
    iterate through data in minibatches as follows:

        for inputs_minibatch in get_minibatches(inputs, minibatch_size):
            ...

    Or with multiple data sources:

        for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
            ...

    Args:
        data: there are two possible values:
            - a list or numpy array
            - a list where each element is either a list or numpy array
        minibatch_size: the maximum number of items in a minibatch
        shuffle: whether to randomize the order of returned data
    Returns:
        minibatches: the return value depends on data:
            - If data is a list/array it yields the next minibatch of data.
            - If data a list of lists/arrays it returns the next minibatch of each element in the
              list. This can be used to iterate through multiple data sources
              (e.g., features and labels) at the same time.

    """
    list_data = type(data) is list and (type(data[0]) is list or
                                        type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start +
                                    minibatch_size]
        yield [minibatch(d, minibatch_indices) for d in data] if list_data \
            else minibatch(data, minibatch_indices)


def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray \
        else [data[i] for i in minibatch_idx]


def minibatches(data, batch_size, shuffle=True):
    """
    return format is a list of [questions, contexts, answers]
    """
    batches = [np.array(col) for col in zip(*data)]
    return get_minibatches(batches, batch_size, shuffle)


def pad_batch_sequences(batch):
    """
    zero padding all sequences to have the length of the longest sequence + 1
    the one extra slot is for the sentinel
    return seq_lens and masks
    """
    seq_lens = [len(seq) for seq in batch]
    max_len = max(seq_lens)
    masks = []
    for seq in batch:
        seq_len = len(seq)
        seq.extend([qa_data.PAD_ID] * (max_len + 1 - seq_len))
        masks.append([True] * seq_len + [False] * (max_len + 1 - seq_len))
    return seq_lens, masks, max_len


class PadInfo(object):
    def __init__(self, cseq_lens, cmasks, cmax_len, qseq_lens, qmasks,
                 qmax_len):
        self.qseq_lens = qseq_lens
        self.qmasks = qmasks
        self.qmax_len = qmax_len
        self.cseq_lens = cseq_lens
        self.cmasks = cmasks
        self.cmax_len = cmax_len

    def print_info(self):
        print 'qseq_lens:', self.qseq_lens
        print 'qmasks:', self.qmasks
        print 'cseq_lens:', self.cseq_lens
        print 'cmasks:', self.cmasks


def pad_minibatched(data, batch_size, shuffle=True):
    """
    batch is [context_batch, question_batch, begin_batch, end_batch]
    """
    data = copy.deepcopy(data)
    batches = minibatches(data, batch_size, shuffle)
    for batch in batches:
        pad_info = PadInfo(*(
            pad_batch_sequences(batch[0]) + pad_batch_sequences(batch[1])))
        batch[0] = np.asarray(batch[0].tolist())
        batch[1] = np.asarray(batch[1].tolist())
        yield batch, pad_info


def read_train_val_data(dir, prefix):
    """
    :return: a list of (context, question, begin, end)
    """
    def line_to_list(line):
        return [int(x) for x in line.split()]

    def gen_data_tuple(i):
        answer = line_to_list(answers[i])
        return (line_to_list(contexts[i]), line_to_list(questions[i]),
                answer[0], answer[1])

    with open(os.path.join(dir, prefix + '.ids.question')) as q_file, \
         open(os.path.join(dir, prefix + '.ids.context')) as c_file, \
         open(os.path.join(dir, prefix + '.span')) as a_file:
        questions = [x.strip() for x in q_file.readlines()]
        contexts = [x.strip() for x in c_file.readlines()]
        answers = [x.strip() for x in a_file.readlines()]
        data = [gen_data_tuple(i) for i in range(len(questions))]
    print '[Done] read data from ' + prefix
    return data


def load_embeddings(embed_path):
    return np.load(embed_path)['glove']


def test_data():
    dir = 'data/squad'
    prefix = 'train'
    data = read_train_val_data(dir, prefix)
    print(len(data))
    batches = pad_minibatched(data, 2)
    batch, pad_info = batches.next()
    context_batch, question_batch, begin_batch, end_batch = batch
    # pad_info.print_info()
