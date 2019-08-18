import numpy as np
import torch

# Inspired from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, filename='checkpoint.pt', patience=7, verbose=False):
        """
        Args:
            filename (str) : where to store the model
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.filename = filename
        self.saving_epoch = 0

    def __call__(self, val_loss, epoch, state):
        """ state is a dictionnary """

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, epoch, state)
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, epoch, state)
            self.counter = 0

    def save_checkpoint(self, val_loss, epoch, state):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'### Validation loss decreased ({self.val_loss_min:.3f} --> {val_loss:.3f}).  Saving model ...')
        state["saving_epoch"] = epoch
        state["val_loss"] = val_loss
        torch.save(state, self.filename)
        self.saving_epoch = epoch
        self.val_loss_min = val_loss