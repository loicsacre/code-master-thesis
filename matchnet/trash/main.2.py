
import errno
import math
import os
import sys
import time
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from scipy import interpolate
from sklearn import metrics
from torch import optim
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm

from dataset_matchnet import CytomineDataset
# from utils import CytomineDataset
from eval_metrics import ErrorRateAt95Recall
from models import MatchNet, transform_matchnet
from models import TransferNet, transform_transfernet

from pt_inspector import WeightMonitor
from pytorchtools import EarlyStopping
from utils import mkdir

DEBUGGING = False

OUTPUT_DIR = "./results/matchnet"

class Config():

    learning_rate = 0.01
    momentum = 0
    weight_decay = 0
    batch_size = 32
    train_number_epochs = 50
    iteration_every_batches = 100
    patience = 7 if not DEBUGGING else 3

    def __init__(self, learning_rate=learning_rate, momentum=momentum, 
        weight_decay=weight_decay, batch_size=batch_size, 
        train_number_epochs=train_number_epochs, 
        iteration_every_batches=iteration_every_batches, 
        patience=patience):
        
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.train_number_epochs = train_number_epochs
        self.iteration_every_batches = iteration_every_batches
        self.patience = patience


config = Config()


class STOP(Exception):
    pass


def get_colorspace_str():
    args = get_args()

    colorspace = "rgb"

    if args.grayscale or args.original:
        colorspace = "gray"
    if args.hsv:
        colorspace = "hsv"

    return colorspace


def get_prefix_file():
    args = get_args()
    colorspace = get_colorspace_str()

    if args.original:
        args.resize = 64

    return f"{args.arch}_{args.size}_{colorspace}_{args.resize}_{config.learning_rate}_{config.momentum}_{config.batch_size}"


def get_ouput_dir_fig():
    args = get_args()
    out_fig = f"{OUTPUT_DIR}/figures/" + args.out
    mkdir(out_fig)
    return out_fig


def save_plot(iteration, loss):

    filename = get_ouput_dir_fig() + "/" + get_prefix_file()

    plt.figure(1, figsize=(10, 8))
    plt.plot(iteration[0], loss[0])
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss training')
    plt.savefig(filename+'-train.png')

    plt.figure(12, figsize=(10, 8))
    plt.plot(iteration[1], loss[1])
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss evaluation')
    plt.savefig(filename+'-eval.png')


def save_checkpoint(state, filename):
    torch.save(state, filename)


def plot_train_val_loss(train_loss, valid_loss, saving_epoch):

    # visualize the loss as the network trained
    fig = plt.figure(num=3, figsize=(10, 8))
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
    plt.ylim(0, max(max(train_loss), max(valid_loss)))  # consistent scale
    plt.xlim(0, len(train_loss)+1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()

    output_dir = get_ouput_dir_fig() + "/loss/"
    mkdir(output_dir)
    fig.savefig(output_dir + get_prefix_file() +
                '-loss_plot.png', bbox_inches='tight')
    fig.savefig(output_dir + get_prefix_file() +
                '-loss_plot.eps', bbox_inches='tight')


def train(model, device, data_args, filename, info, checkpoint=None):

    args = get_args()

    criterion = nn.BCELoss()  # Binary Cross Entropy Loss

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
        avg_train_loss = []  # list of means of training losses for each epoch
        avg_valid_loss = []
    else:
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        counter = checkpoint['counter']
        loss_history = checkpoint['loss_history']
        iteration_number = counter[-1]
        epoch_history = checkpoint['epoch_history']
        time_elapsed = checkpoint['time']
        avg_train_loss = checkpoint['avg_train_loss']
        avg_valid_loss = checkpoint['avg_valid_loss']


    #--------------

    if get_args().arch == "matchnet":
        transform = transform_matchnet
    elif get_args().arch == "transfernet":
        transform = transform_transfernet

    # Training dataset
    train_data = CytomineDataset(dataset_type="training", transform=transform)
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size//2, shuffle=False)

    # Evaluation dataset
    evaluate_data =  CytomineDataset(dataset_type="evaluating", transform=transform)
    evaluate_dataloader = torch.utils.data.DataLoader(
        evaluate_data, batch_size=config.batch_size//2, shuffle=False)

    #--------------

    # # Evaluation dataset
    # evaluate_data = CytomineDataset(dataset_type="evaluating", **data_args)
    # evaluate_dataloader = torch.utils.data.DataLoader(
    #     evaluate_data, batch_size=config.batch_size//2, shuffle=False)

    # # Training dataset
    # if DEBUGGING:
    #     train_data = evaluate_data
    #     train_dataloader = evaluate_dataloader
    # else:
    #     train_data = CytomineDataset(dataset_type="training", **data_args)
    #     train_dataloader = torch.utils.data.DataLoader(
    #         train_data, batch_size=config.batch_size//2, shuffle=False)


    start_time = time.time()

    # Create weight monitor and register model
    # This monitor only works on "in place" tensors, such as those from
    # a module.
    monitor = WeightMonitor().register_model(model)

    print("\n==> Training...")

    try:
        for epoch in range(start_epoch, config.train_number_epochs):

            start_time_epoch = time.time()

            ###################
            # train the model #
            ###################

            model.train()
            train_loss_epoch_list = []
            for batch_idx, (data_sim, data_dissim, target_sim, target_dissim) in enumerate(train_dataloader):

                data = Variable(torch.cat((data_sim, data_dissim))).to(device)
                target = Variable(
                    torch.cat((target_sim, target_dissim))).to(device)

                output = model(data)

                loss = criterion(output, target)

                train_loss = loss.data

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_epoch_list.append(train_loss)

                if batch_idx % config.iteration_every_batches == 0:
                    iteration_number[0] += config.iteration_every_batches
                    counter[0].append(iteration_number[0])
                    loss_history[0].append(train_loss)
                    epoch_history[0].append(epoch)

                if DEBUGGING:
                    break

            ######################
            # Validate the model #
            ######################

            model.eval()
            validation_loss_epoch_list = []
            for batch_idx, (data_sim, data_dissim, target_sim, target_dissim) in enumerate(evaluate_dataloader):

                data = Variable(torch.cat((data_sim, data_dissim))).to(device)
                target = Variable(
                    torch.cat((target_sim, target_dissim))).to(device)

                output = model(data)

                valid_loss = criterion(output, target).data
                validation_loss_epoch_list.append(valid_loss)

                if batch_idx % config.iteration_every_batches == 0:
                    iteration_number[1] += config.iteration_every_batches
                    counter[1].append(iteration_number[1])
                    loss_history[1].append(valid_loss)
                    epoch_history[1].append(epoch)

                if DEBUGGING:
                    break

            # train_loss_epoch_mean = float(
            #     sum(train_loss_epoch_list)/len(train_loss_epoch_list))
            # validation_loss_epoch_mean = float(sum(validation_loss_epoch_list)/len(
            #     validation_loss_epoch_list))

            train_loss_epoch_mean = float(min(train_loss_epoch_list))
            validation_loss_epoch_mean = float(min(validation_loss_epoch_list))

            avg_train_loss.append(train_loss_epoch_mean)
            avg_valid_loss.append(validation_loss_epoch_mean)

            print("Epoch number {}\n Current loss {}/{} (train/eval) ({} sec)".format(
                epoch, train_loss_epoch_mean, validation_loss_epoch_mean, time.time() - start_time_epoch))

            if math.isnan(train_loss_epoch_mean) or math.isnan(validation_loss_epoch_mean):
                print("NAN value")
                # TODO: remove
                print(train_loss_epoch_list)
                print(validation_loss_epoch_list)
                raise STOP

            state = {
                'info': info,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch_history': epoch_history,
                'counter': counter,
                'loss_history': loss_history,
                'time': time.time() - start_time + time_elapsed,
                'avg_train_loss': avg_train_loss,
                'avg_valid_loss': avg_valid_loss
            }

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(round(validation_loss_epoch_mean, 3), epoch, state)

            print("### Analysis at epoch", epoch)
            monitor.analyze()
            print()

            if early_stopping.early_stop:
                print("\nEarly stopping...")
                raise STOP

            if DEBUGGING:
                raise STOP

    except STOP:
        print("Stopping... \nEpoch number {}\n Current loss {}/{} (train/val)\n".format(
            epoch, train_loss_epoch_mean, validation_loss_epoch_mean))

    print("Training time {} sec".format(time.time() - start_time))

    plot_train_val_loss(avg_train_loss, avg_valid_loss,
                        early_stopping.saving_epoch)

    save_plot(counter, loss_history)

    # Evaluate
    evaluate(model, device, data_args, filename,
             dataloader=evaluate_dataloader)


def evaluate(model, device, data_args, filename, dataloader=None, testing=False):

    args = get_args()

    model = get_model(args.arch)

    # Set evaluation mode
    model.eval()

    with torch.no_grad():

        if torch.cuda.is_available():
            checkpoint = torch.load(filename)
        else:
            checkpoint = torch.load(filename, map_location='cpu')

        model.load_state_dict(checkpoint['state_dict'])

        if testing:
            data_args["dataset_type"] = "testing"
            print("\n\n==> Testing..")
        else:
            data_args["dataset_type"] = "evaluating"
            print("\n\n==> Evaluating..")

        if dataloader is None:
            dataset = CytomineDataset(**data_args)

            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=config.batch_size//2, shuffle=False)

        start_time = time.time()

        test_targets = []
        test_outputs = []

        for batch_idx, (data_sim, data_dissim, target_sim, target_dissim) in enumerate(dataloader):

            input_batch = Variable(
                torch.cat((data_sim, data_dissim))).to(device)
            target = Variable(
                torch.cat((target_sim, target_dissim))).to(device)

            output = model(input_batch).data

            test_targets += list(target.cpu().numpy())  # to cpu
            test_outputs += list(output.cpu().numpy())  # to cpu

        print("Elapsed time {} sec".format(time.time() - start_time))

        fpr, tpr, thresholds = metrics.roc_curve(
            test_targets, test_outputs, pos_label=1.0)

        roc_auc = metrics.auc(fpr, tpr)
        roc_auc = round(roc_auc, 1)

        fpr95 = float(interpolate.interp1d(tpr, fpr)(0.95))
        fpr95 = round(fpr95*100, 1)

        error95 = ErrorRateAt95Recall(test_targets, test_outputs)

        # Make the ROC curve with FPR95

        plt.figure(2, figsize=(10, 8))
        plt.plot([0, 1], [0, 1], 'k--')

        colorspace = get_colorspace_str()

        plt.plot(
            fpr, tpr, label=f"{args.arch}_{colorspace} ({fpr95}% - {roc_auc})")
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')

        out_fig = f"{get_ouput_dir_fig()}/ROC"
        mkdir(out_fig)
        filename = out_fig + "/" + get_prefix_file()

        plt.savefig(filename + "-ROC-eval.png")
        plt.savefig(filename + "-ROC-eval.eps")

        print('FPR95:', fpr95)
        print('Error95:', error95)


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

    parser.add_argument('-g', dest='grayscale',
                        action="store_true",
                        help="Indicates whether if the patches must be converted in gray scale")

    parser.add_argument('-o', dest='original',
                        action="store_true",
                        help="Indicates whether the model is trained with original settings (same as papers)")

    parser.add_argument('-hsv', dest='hsv',
                        action="store_true",
                        help="Indicates whether if the patches must be converted in hsv colorspace")

    parser.add_argument('--size', dest='size',
                        default=300,
                        type=int,
                        help="Size of the crops for the training")

    parser.add_argument("--arch", dest='arch',
                        default='matchnet',
                        choices=["matchnet"],
                        help="model architecture: \
                              matchnet (MatchNet) | \
                             (default: matchnet)")

    parser.add_argument("--resize", dest='resize',
                        default=None,
                        type=int,
                        help="Resize the patches if necessary")

    parser.add_argument('--out', dest='out',
                        default='',
                        help="Where the checkpoints will be stored")

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

    parser.add_argument('-d', dest='d',
                        action="store_true",
                        help="Activate debug mode")

    return parser.parse_args()


def get_model(arch):

    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ch_in = 3
    # if the gray scale mode is activated, only one channel needed
    if args.grayscale:
        ch_in = 1

    if args.resize is not None:
        params = {"size": int(args.resize), "ch_in": ch_in}
    else:
        params = {"size": int(args.size), "ch_in": ch_in}

    if not args.original:
        # TODO remove
        if arch == "matchnet":
            model = MatchNet()
        pass
        # if arch == "matchnet":
        #     model = MatchNet
        #     model = model(**params)
    else:
        if arch == "matchnet":
            model = MatchNet()

    model.to(device)
    return model


def init():

    args = get_args()

    config.learning_rate = args.learning_rate
    config.momentum = args.momentum
    config.batch_size = args.batch_size

    global DEBUGGING
    DEBUGGING = args.d


def main():
    """main function"""

    args = get_args()

    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # If original settings, work with grayscale images and 64x64 patches
    if args.original:
        args.resize = 64
        args.grayscale = True

    model = get_model(args.arch)

    print("==> Using architecture : {}".format(args.arch), type(model))

    config_dict = dict([(name, value) for name, value in vars(
        config).items() if not name.startswith('_')])

    print("==> Config:")

    for (n, v) in config_dict.items():
        print(" ", n, ":", v)

    nb_of_transform = 7

    data_args = {"size": args.size,
                 "resize": args.resize,
                 "colorspace": get_colorspace_str(),
                 "nb_of_transform": nb_of_transform,
                 }

    info = {
        "args": vars(args),
        "nb_transform": nb_of_transform,
        "config": config
    }

    output_dir_checkpoint = os.path.join(OUTPUT_DIR, "checkpoints", args.out)
    mkdir(output_dir_checkpoint)
    filename_checkpoint = f"{output_dir_checkpoint}/{get_prefix_file()}.check"

    # Training
    if not args.testing:

        # Set training mode
        model.train()

        checkpoint = None
        if args.resume:
            if os.path.isfile(filename_checkpoint):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(filename_checkpoint)
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(filename_checkpoint, checkpoint['epoch']))
            else:
                print("=> no checkpoint found for '{}'".format(
                    filename_checkpoint))

        if not checkpoint:
            train(model, device, data_args, filename_checkpoint, info)
        else:
            train(model, device, data_args, filename_checkpoint,
                  info, checkpoint)

    # Testing
    else:
        if os.path.isfile(filename_checkpoint):
            evaluate(model, device, data_args,
                     filename_checkpoint, testing=True)
        else:
            print("Checkpoint does not exist..")


if __name__ == "__main__":
    init()

    if not DEBUGGING:
        output_sdout_tmp = f"{OUTPUT_DIR}/outputs/output-{get_prefix_file()}"
        mkdir(os.path.dirname(output_sdout_tmp))

        counter = 1
        output_sdout = output_sdout_tmp
        while os.path.exists(output_sdout + ".txt"):
            output_sdout = f"{output_sdout_tmp}-{counter}"
            counter += 1
        sys.stdout = open(output_sdout + ".txt", 'w')

    main()
