
import errno
import math
import os
import time
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch
from scipy import interpolate
from sklearn import metrics
from torch import optim
from torch.autograd import Variable

from dataset_siamese import CytomineDataset
from models import SiameseAlexNet
from training_tools import (AdaptiveTransformation, EarlyStopping,
                            ErrorRateAt95Recall, TripletLoss)
from training_tools.pt_inspector import WeightMonitor
from utils import mkdir
from torchvision import transforms

DEBUGGING = False

transform_composition = []
num_output_channels = 3
transform_composition.append(transforms.Grayscale(
    num_output_channels=num_output_channels))
transform_composition.append(transforms.ToTensor())
transform_composition.append(transforms.Normalize(
    [0.5]*num_output_channels, [0.625]*num_output_channels))
TRANSFORM = transforms.Compose(transform_composition)


class Config():

    learning_rate = 0.01
    momentum = 0
    weight_decay = 0
    batch_size = 32
    train_number_epochs = 50
    iteration_every_batches = 100
    margin = 1.0
    patience = 7 if not DEBUGGING else 3
    stop_iteration = 2

    def __init__(self, learning_rate=learning_rate, momentum=momentum,
                 weight_decay=weight_decay, batch_size=batch_size,
                 train_number_epochs=train_number_epochs, margin=margin,
                 iteration_every_batches=iteration_every_batches,
                 patience=patience):

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.train_number_epochs = train_number_epochs
        self.margin = margin
        self.iteration_every_batches = iteration_every_batches
        self.patience = patience


config = Config()


class STOP(Exception):
    pass


def get_prefix_file():
    args = get_args()
    return f"{args.arch}_{config.learning_rate}_{config.momentum}_{config.batch_size}-{args.job_id}"


def get_ouput_dir_fig(testing=False):
    args = get_args()
    if not testing:
        out_fig = f"{args.output_dir}/{args.arch}/training/figures/"
    else:
        out_fig = f"{args.output_dir}/{args.arch}/testing/figures/"

    mkdir(out_fig)
    return out_fig


def save_plot_iteration_loss(iteration, loss):

    file_dir = get_ouput_dir_fig() + "iterations/"
    mkdir(file_dir)
    filename = file_dir + get_prefix_file()

    plt.figure(plt.gcf().number+1, figsize=(10, 8))
    plt.plot(iteration[0], loss[0])
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss training')
    plt.savefig(filename + '-train.png')

    plt.figure(plt.gcf().number+1, figsize=(10, 8))
    plt.plot(iteration[1], loss[1])
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss evaluation')
    plt.savefig(filename + '-eval.png')


def save_checkpoint(state, filename):
    torch.save(state, filename)


def plot_train_val_loss(train_loss, valid_loss, saving_epoch):

    # visualize the loss as the network trained
    fig = plt.figure(num=plt.gcf().number+1, figsize=(10, 8))
    plt.plot(range(1, len(train_loss)+1), train_loss, label='Training Loss')
    plt.plot(range(1, len(valid_loss)+1), valid_loss, label='Validation Loss')

    # find position of lowest validation loss
    # minposs = valid_loss.index(min(valid_loss))+1
    # do +1 for the plot (0 is the index of first epoch)
    minposs = saving_epoch + 1
    plt.axvline(minposs, linestyle='--', color='r',
                label='Early Stopping Checkpoint')

    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.ylim(0, max(train_loss[0], valid_loss[0]))  # consistent scale
    plt.xlim(0, len(train_loss)+1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()

    output_dir_fig = get_ouput_dir_fig() + "/loss/"
    mkdir(output_dir_fig)
    fig.savefig(output_dir_fig + get_prefix_file() +
                '-loss_plot.png', bbox_inches='tight')
    fig.savefig(output_dir_fig + get_prefix_file() +
                '-loss_plot.eps', bbox_inches='tight')


def train(model, filename, info, checkpoint=None):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    if torch.cuda.is_available():
        print("## CUDA available")
        print(f"Current device: {torch.cuda.current_device()}")
        print(
            f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # nn.BCELoss()  # Binary Cross Entropy Loss
    criterion = TripletLoss(config.margin)

    optimizer = optim.SGD(model.parameters(),
                          momentum=config.momentum,
                          lr=config.learning_rate,
                          weight_decay=config.weight_decay)

    # optimizer = optim.Adam(model.parameters(),
    #                       lr=config.learning_rate,
    #                       weight_decay=config.weight_decay)

    print("==> Loss", criterion)
    print("==> Optimizer", optimizer)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(
        filename=filename, patience=config.patience, verbose=True)

    if not checkpoint:
        start_epoch = 0
        counter = [[], []]
        loss_history = [[], []]
        epoch_history = [[], []]
        iteration_number = [0]*2
        time_elapsed = 0
        train_loss_history = []  # list of means of training losses for each epoch
        val_loss_history = []
    else:
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        counter = checkpoint['counter']
        loss_history = checkpoint['loss_history']
        iteration_number = [counter[0][-1], counter[1][-1]]
        epoch_history = checkpoint['epoch_history']
        time_elapsed = checkpoint['time']
        train_loss_history = checkpoint['train_loss_history']
        val_loss_history = checkpoint['val_loss_history']

    transform = None
    adaptative_transform = AdaptiveTransformation()

    # Training dataset
    train_data = CytomineDataset(dataset_type="training", transform=transform, adaptative_transform=adaptative_transform,
                                 batch_size=get_args().batch_size, verbose=get_args().v)
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=get_args().num_workers)

    # Evaluation dataset
    evaluate_data = CytomineDataset(
        dataset_type="evaluating", transform=transform, adaptative_transform=adaptative_transform, batch_size=get_args().batch_size, verbose=get_args().v)
    evaluate_dataloader = torch.utils.data.DataLoader(
        evaluate_data, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=get_args().num_workers)

    start_time = time.time()

    # Create weight monitor and register model
    # This monitor only works on "in place" tensors, such as those from
    # a module.
    monitor = WeightMonitor().register_model(model)

    print("\n###################")
    print("### Training... ###")
    print("###################\n")

    train_loss_epoch = 0
    val_loss_epoch = 0

    try:
        for epoch in range(start_epoch, config.train_number_epochs):

            print(f"\n-------------------------")
            print(f"---- Epoch number {epoch} ----")
            print(f"-------------------------\n")

            start_time_epoch = time.time()

            ###################
            # train the model #
            ###################

            model.train()
            train_loss_epoch_list = []

            start_time_t = time.time()
            total_it = 0

            for batch_idx, (anchor, positive, negative) in enumerate(train_dataloader):

                start_time_it = time.time()

                anchor = Variable(anchor).to(device)
                positive = Variable(positive).to(device)
                negative = Variable(negative).to(device)

                anchor_o, positive_o, negative_o = model.forward_triplet(
                    anchor, positive, negative)
                loss = criterion(anchor_o, positive_o, negative_o)

                train_loss = loss.data

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_epoch_list.append(train_loss)

                # if batch_idx % config.iteration_every_batches == 0:
                iteration_number[0] += 1  # config.iteration_every_batches
                counter[0].append(iteration_number[0])
                loss_history[0].append(train_loss)
                epoch_history[0].append(epoch)

                total_it += (time.time() - start_time_it)

                if get_args().v:
                    print(
                        f"* Iteration {batch_idx} : {(time.time() - start_time_it):3f} s (total: {total_it:3f})")

                if DEBUGGING and batch_idx == (config.stop_iteration-1):
                    break

            print(
                f"** Training (epoch {epoch}) : {(time.time() - start_time_t):3f} s (total it : {total_it:3f})")

            ######################
            # Validate the model #
            ######################

            model.eval()
            validation_loss_epoch_list = []
            for batch_idx, (anchor, positive, negative) in enumerate(evaluate_dataloader):

                anchor = Variable(anchor).to(device)
                positive = Variable(positive).to(device)
                negative = Variable(negative).to(device)

                anchor_o, positive_o, negative_o = model.forward_triplet(
                    anchor, positive, negative)

                valid_loss = criterion(anchor_o, positive_o, negative_o).data
                validation_loss_epoch_list.append(valid_loss)

                # if batch_idx % config.iteration_every_batches == 0:
                iteration_number[1] += 1  # config.iteration_every_batches
                counter[1].append(iteration_number[1])
                loss_history[1].append(valid_loss)
                epoch_history[1].append(epoch)

                if DEBUGGING and batch_idx == (config.stop_iteration-1):
                    break

            train_loss_epoch = float(min(train_loss_epoch_list))
            val_loss_epoch = float(min(validation_loss_epoch_list))

            train_loss_history.append(train_loss_epoch)
            val_loss_history.append(val_loss_epoch)

            print(
                f"## Current loss {train_loss_epoch:3f}/{val_loss_epoch:3f} (train/eval) ({(time.time() - start_time_epoch):3f} sec)")

            if math.isnan(train_loss_epoch) or math.isnan(val_loss_epoch):
                print("!!! NAN values !!!")
                raise STOP

            state = {
                'info': info,
                'transform': transform,
                'adaptative_transform': adaptative_transform,
                'early_stopping_saving_epoch': early_stopping.saving_epoch,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch_history': epoch_history,
                'counter': counter,
                'loss_history': loss_history,
                'time': time.time() - start_time + time_elapsed,
                'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history
            }

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(round(val_loss_epoch, 3), epoch, state)

            print("## Analysis")
            monitor.analyze()
            print()

            # Additional info when using cuda
            if get_args().v and device.type == 'cuda':
                print(torch.cuda.get_device_name(0))
                print('Memory Usage:')
                print(' - Allocated:',
                      round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
                print(' - Cached:   ',
                      round(torch.cuda.memory_cached(0)/1024**3, 1), 'GB')

            if early_stopping.early_stop:
                print("\nEarly stopping...")
                raise STOP

            if DEBUGGING:
                raise STOP

    except STOP:
        print("=============================")
        print(f"Stopping on epoch {epoch}")
        print(
            f"Current loss {train_loss_epoch:3f}/{val_loss_epoch:3f} (train/eval)")
        print(f"Training time {(time.time() - start_time):3f} sec")
        print("=============================")

        plot_train_val_loss(train_loss_history, val_loss_history,
                            early_stopping.saving_epoch)

        save_plot_iteration_loss(counter, loss_history)

        # Evaluate
        evaluate(model, filename)

        # Testing
        evaluate(model, filename, testing=True)
        return
    print("\nReached the maximum number of epochs...")


def evaluate(model, filename_checkpoint, dataloader=None, testing=False):

    dataset_type = "evaluating"
    if testing:
        print("\n###################")
        print("### Testing... ####")
        print("###################\n")
        dataset_type = "testing"
    else:
        print("\n###################")
        print("## Evaluating... ##")
        print("###################\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = get_args()

    model = get_model(args.arch)

    # Set evaluation mode
    model.eval()

    with torch.no_grad():

        if torch.cuda.is_available():
            checkpoint = torch.load(filename_checkpoint)
        else:
            checkpoint = torch.load(filename_checkpoint, map_location='cpu')

        model.load_state_dict(checkpoint['state_dict'])

        if dataloader is None:
            transform = None
            adaptative_transform = AdaptiveTransformation()

            data = CytomineDataset(
                dataset_type=dataset_type, transform=transform, adaptative_transform=adaptative_transform, batch_size=get_args().batch_size, verbose=get_args().v)
            dataloader = torch.utils.data.DataLoader(
                data, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=get_args().num_workers)

        start_time = time.time()

        test_targets = []
        test_outputs = []

        for batch_idx, (anchor, positive, negative) in enumerate(dataloader):

            anchor = Variable(anchor).to(device)
            positive = Variable(positive).to(device)
            negative = Variable(negative).to(device)

            number_of_samples = anchor.size()[0]

            anchor_o, positive_o, negative_o = model.forward_triplet(
                anchor, positive, negative)

            distance_positive = (anchor_o - positive_o).pow(2).sum(1)
            test_targets += list([0]*number_of_samples)  # to cpu
            test_outputs += list(distance_positive.cpu().numpy())  # to cpu

            distance_negative = (anchor_o - negative_o).pow(2).sum(1)
            test_targets += list([1]*number_of_samples)  # to cpu
            test_outputs += list(distance_negative.cpu().numpy())  # to cpu

            if DEBUGGING:
                break

        print("Elapsed time {} sec".format(time.time() - start_time))

        fpr, tpr, thresholds = metrics.roc_curve(
            test_targets, test_outputs, pos_label=1.0)

        roc_auc = metrics.auc(fpr, tpr)
        roc_auc = round(roc_auc, 2)

        fpr95 = float(interpolate.interp1d(tpr, fpr)(0.95))
        fpr95 = round(fpr95*100, 1)

        error95 = ErrorRateAt95Recall(test_targets, test_outputs)

        # Make the ROC curve with FPR95

        plt.figure(plt.gcf().number+1, figsize=(10, 8))
        plt.plot([0, 1], [0, 1], 'k--')

        plt.plot(
            fpr, tpr, label=f"{args.arch} ({fpr95}% - {roc_auc})")
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')

        out_fig = f"{get_ouput_dir_fig(testing)}/ROC"

        mkdir(out_fig)
        filename = out_fig + "/" + \
            filename_checkpoint.split("/")[-1].rsplit(".", 1)[0]
        plt.savefig(filename + f"-ROC-{dataset_type}.png")
        plt.savefig(filename + f"-ROC-{dataset_type}.eps")

        print(f"AUC: {roc_auc}")
        print(f"FPR95: {fpr95}")
        print(f"Error95: {error95}")
        print(f"Thresholds: {thresholds}")


def get_args():
    """get the arguments of the program"""
    parser = ArgumentParser(
        prog="Program to train a different types of networks for patch similarity")

    parser.add_argument('-t', dest='testing',
                        action="store_true",
                        help="Indicates whether it is for testing")

    parser.add_argument('-r', dest='resume',
                        action="store_true",
                        help="Indicates whether the training must be resumed from some checkpoint")

    parser.add_argument("--arch", dest='arch',
                        default='siameseAlexnet',
                        choices=["siameseAlexnet"],
                        help="model architecture: \
                              siameseAlexnet (SiameseAlexNet) | \
                             (default: siameseAlexnet)")

    parser.add_argument('--output_dir', dest='output_dir',
                        default="./results/siamese",
                        help="Where all the results will be saved (checkpoints and figures)")

    parser.add_argument('--checkpoint', dest='checkpoint',
                        help="Path to the checkpoint")

    parser.add_argument('--learning_rate', dest='learning_rate',
                        default=config.learning_rate,
                        type=float,
                        help="The learning rate for SGD")

    parser.add_argument('--momentum', dest='momentum',
                        default=config.momentum,
                        type=float,
                        help="The momentum for SGD")

    parser.add_argument('--batch_size', dest='batch_size',
                        default=config.batch_size,
                        type=int,
                        help="The batch size")

    parser.add_argument('--margin', dest='margin',
                        default=config.margin,
                        type=float,
                        help="The margin for the triplet loss")

    parser.add_argument('--num-workers', dest="num_workers",
                        default=2, type=int)

    parser.add_argument('-d', dest='d',
                        action="store_true",
                        help="Activate debug mode")

    parser.add_argument('-v', dest='v',
                        action="store_true",
                        help="Activate verbose for the dataloader and GPU usage")

    parser.add_argument('--job_id', dest='job_id',
                        default=0,
                        type=int,
                        help="The job id")

    return parser.parse_args()


def get_model(arch):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"## Using device {device.type}")

    model = SiameseAlexNet()

    if arch == "siameseAlexnet":
        model = SiameseAlexNet()

    model.to(device)
    return model


def init():

    args = get_args()

    config.learning_rate = args.learning_rate
    config.momentum = args.momentum
    config.batch_size = args.batch_size
    config.margin = args.margin

    global DEBUGGING
    DEBUGGING = args.d


def main():
    """main function"""

    args = get_args()
    mkdir(args.output_dir)

    print("\n#################")
    print("### Arguments ###")
    print("#################")
    for arg in vars(args):
        print(f"{arg} : {getattr(args, arg)}")
    print("#################\n")

    model = get_model(args.arch)

    print("==> Using architecture : {}".format(args.arch), type(model))

    config_dict = dict([(name, value) for name, value in vars(
        config).items() if not name.startswith('_')])

    print("==> Config:")

    for (n, v) in config_dict.items():
        print(" ", n, ":", v)

    info = {
        "args": vars(args),
        "config": list(config_dict.items())
    }

    output_dir_checkpoint = os.path.join(
        args.output_dir, args.arch, "checkpoints")
    mkdir(output_dir_checkpoint)
    filename_checkpoint = f"{output_dir_checkpoint}/{get_prefix_file()}.check"

    # Training
    if not args.testing:

        # Set training mode
        model.train()

        print("--->", model.training)
        return

        checkpoint = None
        if args.resume:

            print("\n###################")
            print("### Resuming... ###")
            print("###################\n")

            if os.path.isfile(args.checkpoint):
                print(f"=> loading checkpoint '{args.resume}'")
                if torch.cuda.is_available():
                    checkpoint = torch.load(args.checkpoint)
                else:
                    checkpoint = torch.load(
                        args.checkpoint, map_location='cpu')
                print(
                    f"=> loaded checkpoint '{args.checkpoint}' (epoch {checkpoint['epoch']})\n")
            else:
                print(f"=> no checkpoint found for '{args.checkpoint}'\n")
                return

        if checkpoint is None:
            train(model, filename_checkpoint, info)
        else:
            train(model, filename_checkpoint, info, checkpoint)

    # Testing
    else:
        if os.path.isfile(args.checkpoint):
            evaluate(model, args.checkpoint, testing=True)
        else:
            print("Checkpoint does not exist..")


if __name__ == "__main__":
    init()
    main()
