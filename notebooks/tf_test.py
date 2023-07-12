import numpy as np
import sys
import os
import pickle
import copy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

tf.config.run_functions_eagerly(True)

class SplitMnistGenerator():
    def __init__(self):

        # Transform to normalized Tensors 
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST('data', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST('data', train=False, transform=transform, download=True)
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        self.X_train = next(iter(train_loader))[0].numpy().reshape(-1, 28*28)
        self.X_test = next(iter(test_loader))[0].numpy().reshape(-1, 28*28)
        self.train_label = next(iter(train_loader))[1].numpy()
        self.test_label = next(iter(test_loader))[1].numpy()

        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]
        self.max_iter = len(self.sets_0)
        self.cur_iter = 0

    def get_dims(self):
        return self.X_train.shape[1], 2

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            # Retrieve train data
            train_0_id = np.where(self.train_label == self.sets_0[self.cur_iter])[0]
            train_1_id = np.where(self.train_label == self.sets_1[self.cur_iter])[0]
            next_x_train = np.vstack((self.X_train[train_0_id], self.X_train[train_1_id]))

            next_y_train = np.hstack(
                (np.zeros((train_0_id.shape[0])), np.ones((train_1_id.shape[0])))
            )

            # Retrieve test data
            test_0_id = np.where(self.test_label == self.sets_0[self.cur_iter])[0]
            test_1_id = np.where(self.test_label == self.sets_1[self.cur_iter])[0]
            next_x_test = np.vstack((self.X_test[test_0_id], self.X_test[test_1_id]))

            next_y_test = np.hstack(
                (np.zeros((test_0_id.shape[0])), np.ones((test_1_id.shape[0])))
            )

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

class ListMLP(tf.Module):
    def __init__(self, num_tasks, input_dim, output_size, hidden_dim, single_head):
        super().__init__()
        self.single_head = single_head
        if self.single_head:
            self.num_tasks = 1
        else:
            self.num_tasks = num_tasks

        self.MLPs = [MLP(input_dim=input_dim, output_size=output_size, hidden_dim=hidden_dim) for _ in range(self.num_tasks)]
        
    def __call__(self, task_id, x):
        if self.single_head:
            model = self.MLPs[0]
        else:
            model = self.MLPs[task_id]
        return model(x)

class MLP(tf.Module):
    def __init__(self, input_dim, output_size, hidden_dim):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(output_size, activation=None)
    
    def __call__(self, x):
        x = self.dense1(x)
        return self.dense2(x) 

@tf.function
def train(model, task_id, loss_fnc, optimizer, train_loader, no_epochs):

    one_hot = tf.keras.layers.CategoryEncoding(num_tokens=2, output_mode="one_hot")
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    for epoch in range(no_epochs):

        #print([m.trainable_variables for m in model.MLPs])
        # Single model [(28*28 x 256), (256,) (256 x 2), (2,)]
        # multiple models [(28*28 x 256), (256,) (256 x 2), (2,)] x num_tasks

        for data, target in train_loader:
            train_loss.reset_states()
            with tf.GradientTape() as tape:
                # training=True is only needed if there are layers with different
                # behavior during training versus inference (e.g. Dropout).
                predictions = model(task_id, data)
                loss = loss_fnc(one_hot(target), predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            # previous gradients are None, so we need to replace them with zeros
            gradients = [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(
                    model.trainable_variables, gradients)]
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss.update_state(loss)

        print(
            'Train Epoch: [{}'.format(epoch),
            '/ {}]'.format(no_epochs),
            '\tLoss: {}'.format(train_loss.result()),
        )

@tf.function
def test(model, task_id, loss_fnc, test_loader):

    one_hot = tf.keras.layers.CategoryEncoding(num_tokens=2, output_mode="one_hot")

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    for data, target in test_loader:
        test_accuracy.reset_states()
        test_loss.reset_states()
        output = model(task_id, data)
        loss = loss_fnc(one_hot(target), output)
        test_loss.update_state(loss)
        test_accuracy.update_state(target, output)

    print(
        'test loss: {}'.format(test_loss.result()),
        'test accuracy: {}'.format(test_accuracy.result()),
    )    
    return test_accuracy.result()

def scores_to_arr(all_acc, acc):
    if all_acc.size == 0:
        all_acc = np.reshape(acc, (1,-1))
    else:
        new_arr = np.empty((all_acc.shape[0], all_acc.shape[1]+1))
        new_arr[:] = np.nan
        new_arr[:,:-1] = all_acc
        all_acc = np.vstack((new_arr, acc))
    return all_acc

def run(no_epochs, data_gen, seed, single_head):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    in_dim, out_dim = data_gen.get_dims()
    test_loaders = []

    all_acc = np.array([])
    
    model_list = ListMLP(num_tasks=data_gen.max_iter, input_dim=28*28, output_size=2, hidden_dim=256, single_head=single_head)

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    for task_id in range(data_gen.max_iter):
        print("Task {}".format(task_id))
        x_train, y_train, x_test, y_test = data_gen.next_task()

        train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
        test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
        test_loaders.append(test_loader)
        
        train(model_list, task_id, loss, optimizer, train_loader, no_epochs) 
        acc = test(model_list, task_id, loss, test_loader)

        accs = []
        for i, test_loader in enumerate(test_loaders): 
           acc = test(model_list, i, loss, test_loader)
           accs.append(acc)
        all_acc = scores_to_arr(all_acc, accs)

    return all_acc

n_runs = 1
n_tasks = 5
epochs = 10
single_head = False
results = np.zeros((n_runs, n_tasks, n_tasks))
for i in range(n_runs):
    seed = 101 + i
    data_gen = SplitMnistGenerator()
    accs = run(epochs, data_gen, seed, single_head)
    results[i, :, :] = accs 
print(results)