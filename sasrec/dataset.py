import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue


def check_index(user_item_matrix):
    min_user_idx = user_item_matrix[:, 0].min()
    if min_user_idx == 0:
        user_item_matrix[:,0] = user_item_matrix[:,0] + 1
    elif min_user_idx >= 2: 
        raise ValueError("Unusual User Index")

    min_item_idx = user_item_matrix[:, 1].min()
    if min_item_idx == 0:
        user_item_matrix[:,1] = user_item_matrix[:,1] + 1
    elif min_item_idx >= 2: 
        raise ValueError("Unusual Item Index")

    return user_item_matrix


def build_index(data_dir, dataset_name):
    user_item_matrix = np.loadtxt(f'{data_dir}/{dataset_name}.txt', dtype=np.int32)
    user_item_matrix = check_index(user_item_matrix)


    print(f"User Indices: {user_item_matrix[:,0].min()}~{user_item_matrix[:,0].max()} -> Total {user_item_matrix[:,0].max()-user_item_matrix[:,0].min()+1}")
    print(f"Item Indices: {user_item_matrix[:,1].min()}~{user_item_matrix[:,1].max()} -> Total {user_item_matrix[:,1].max()-user_item_matrix[:,0].min()+1}")

    num_users = user_item_matrix[:, 0].max()
    num_items = user_item_matrix[:, 1].max()


    item_seqs = [[] for _ in range(num_users+1)]
    user_seqs = [[] for _ in range(num_items+1)]


    for user, item in user_item_matrix:
        item_seqs[user].append(item)
        user_seqs[item].append(user)

    return item_seqs, user_seqs


def data_partition(data_dir, dataset_name):
    users_item_seqs_dict = defaultdict(list)
    item_seqs_train = {}
    item_seqs_valid = {}
    item_seqs_test = {}

    user_item_matrix = np.loadtxt(f'{data_dir}/{dataset_name}.txt', dtype=np.int32)
    user_item_matrix = check_index(user_item_matrix)
    print("assume user/item index starting from 1")

    num_users = user_item_matrix[:, 0].max()
    num_items = user_item_matrix[:, 1].max()

    for u,i in user_item_matrix:
        users_item_seqs_dict[u].append(i)

    for user_idx in users_item_seqs_dict:
        nfeedback = len(users_item_seqs_dict[user_idx])
        if nfeedback < 3:
            item_seqs_train[user_idx] = users_item_seqs_dict[user_idx]
            item_seqs_valid[user_idx] = []
            item_seqs_test[user_idx] = []
        else:
            item_seqs_train[user_idx] = users_item_seqs_dict[user_idx][:-2]
            item_seqs_valid[user_idx] = []
            item_seqs_valid[user_idx].append(users_item_seqs_dict[user_idx][-2])
            item_seqs_test[user_idx] = []
            item_seqs_test[user_idx].append(users_item_seqs_dict[user_idx][-1])

    return [item_seqs_train, item_seqs_valid, item_seqs_test, num_users, num_items]


class WarpSampler(object):
    def __init__(self, item_seqs_train, num_users, num_items, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for _ in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(item_seqs_train,
                                                      num_users,
                                                      num_items,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

def random_neq(low, high, positive_set):
    t = np.random.randint(low, high)
    while t in positive_set:
        t = np.random.randint(low, high)
    return t

def sample_function(item_seqs_train, num_users, num_items, batch_size, maxlen, result_queue, random_seed):
    def sample(uid):

        while len(item_seqs_train[uid]) <= 1:
            uid = np.random.randint(num_users)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)

        nxt = item_seqs_train[uid][-1]
        idx = maxlen - 1
        positive_set = set(item_seqs_train[uid])

        for i in reversed(item_seqs_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neq(1, num_items+1, positive_set)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    np.random.seed(random_seed)
    uids = np.arange(1, num_users+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % num_users == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % num_users]))
            counter += 1
        result_queue.put(zip(*one_batch))
