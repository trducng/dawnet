# Helper function during training
# @author: _john
# =============================================================================
import abc
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn


def save_model(session_name, training_iteration, model, optimizer):
    """Save the model"""

    state = {
        'session_name': session_name,
        'training_iteration': training_iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    torch.save(
        state,
        'save/{}_{}.john'.format(session_name, training_iteration))


class BaseJob(abc.ABC):

    def __init__(self, model, criterion, optimizer):
        """Initilize the session"""
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def find_lr(self):
        """Find the suitable learning rate"""
        self.model.train()

        learning_rate = 1e-6
        optimizer = self.optimizer(self.model.parameters(), lr=learning_rate)
        criterion = self.criterion()

        summary = {}
        training_iteration = 0
        accuracy_list = []

        while True:
            if training_iteration >= 100:
                accuracy = sum(accuracy_list) / len(accuracy_list)
                summary[learning_rate] = accuracy

                if learning_rate > 10:
                    # maximum learning rate to test is 10
                    break

                learning_rate *= 2
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate

                training_iteration = 0
                accuracy_list = []

            training_iteration += 1

            X, y = self.get_batch(data_type=1)
            cost = criterion(self.model(X), y)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            self.model.eval()
            correct = 0
            total = 0
            test_folders = glob.glob('data/test/*_*')
            for each_test_folder in test_folders:
                X, y = self.get_batch(data_type=2)
                preds = self.model(X)
                preds = preds.cpu().data.numpy()
                top_1 = np.argmax(preds, axis=1)
                correct += np.sum((top_1 == y).astype(np.int32))
                total += len(y)

            accuracy_list.append(correct / total)
            self.model.train()
        
        # Obtain the best accuracy
        best_lr = None
        best_accuracy = float('-inf')
        for lr, accuracy in summary.items():
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_lr = lr

        print('Will use learning rate value of', best_lr)
        return best_lr

    def restart(self):
        """Restart the session"""
        self.model.restart()

    @abc.abstractmethod
    def get_batch(self, data_type):
        """Get a batch of data"""
        pass

class ImageClassificationJob(BaseJob):
    """A training job

    @TODO: make the training also friendly in JupyterNotebook
        (ability to show visualization, constant reporting)

    # Arguments:
        model [PyTorch Module object]: the model
        project_folder [string]: the parent project folder
    """
    def __init__(self, model, criterion=None, project_folder='', **kwargs):
        """Initialize the training job"""
        self.model = model
        self.optimizer = None
        self.scheduler = None       # actually there are 2 schedulers worth trying
        self.criterion = None
        self.callbacks = []

        # Training hyperparameters
        self.lr = kwargs.get('lr', None)

        # Report
        self.info = {}

        # State variables
        self.initialized = False
        self.training_iteration = 0

    def initialize(self, load=None):
        """Initialize the training session"""

        if self.criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        
        if self.lr is None:
            lr = self.find_lr()

    def fit(self):
        """Perform the training

        More specifically, this method:
            1. trains model
            2. creates checkpoint
            3. reports progress
            4. saves information for review
        """
        while True:
            self.scheduler.step()
            self.training_iteration += 1

            X, y = self.get_batch()
            preds = self.model(X)
            cost = self.criterion(preds, y)

            self.optimizer.zero_grad()
            cost.backward()
            self.optimizer.step()

    def eval(self):
        """Perform the evaluation"""
        pass

    def test(self):
        """Perform model testing"""
        pass

