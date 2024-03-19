import time
import os
import torch
from network import FPNN, FPNN_wo_pp, FPNN_imgonly, FPNN_wo_mask
import argparse
import numpy as np
import matplotlib.pyplot as plt
import copy
from torch.utils.data import DataLoader
from utils import loss_fn, model_evaluation
from dataset_upload import CustomDataset


def main(args):
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_set = CustomDataset(args.train_image_path, args.train_mask_path, args.train_pick_dot_path,
                              args.train_place_dot_path, args.train_csv_path, 'train')
    test_set = CustomDataset(args.test_image_path, args.test_mask_path, args.test_pick_dot_path,
                             args.test_place_dot_path, args.test_csv_path, 'test')

    train_loader = DataLoader(
        dataset=train_set, batch_size=args.train_batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(
        dataset=test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
    fpnn = FPNN().to(device)
    parameters = filter(lambda p: p.requires_grad, fpnn.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    # start training
    best_ap = 100.0
    train_loss_over_epochs = []
    test_ap_over_epochs = []
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    print("Dataset is ready. Start training...")
    for epoch in range(args.epochs):
        print("Current learning rate: ", optimizer.param_groups[0]['lr'])
        fpnn.train()
        running_loss = 0.0
        for iteration, (image, mask, pick_dot, place_dot, label) in enumerate(train_loader):
            image = image.to(device)
            mask = mask.to(device)
            pick_dot = pick_dot.to(device)
            place_dot = place_dot.to(device)
            label = label.to(device)
            pred = fpnn(image, mask, pick_dot, place_dot)
            # pred = fpnn(image, mask)
            # pred = fpnn(image, pick_dot, place_dot)
            # pred = fpnn(image)
            loss = loss_fn(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        running_loss /= len(train_loader)
        print("Epoch {:02d}/{:02d}, Loss {:9.4f}".format(
            epoch + 1, args.epochs, running_loss))

        ap = model_evaluation(test_loader, fpnn, device)
        print("Loss on the test set: {:.4f}".format(ap))

        train_loss_over_epochs.append(running_loss)
        test_ap_over_epochs.append(ap)

        # save the best model
        if ap < best_ap:
            PATH = './CNN_model.pth'
            torch.save(fpnn.state_dict(), PATH)
            print('model is improved and saved')
            best_ap = copy.deepcopy(ap)
        scheduler.step()
    fig = plt.figure()

    plt.subplot(2, 1, 1)
    plt.ylabel("Train loss")
    plt.plot(np.arange(args.epochs) + 1, train_loss_over_epochs, 'k-')
    plt.title("Train loss and test average precision")
    plt.xlim(1, args.epochs)
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.ylabel("Test average precision")
    plt.plot(np.arange(args.epochs) + 1, test_ap_over_epochs, 'b-')
    plt.xlabel("Epochs")
    plt.xlim(1, args.epochs)
    plt.grid(True)
    plt.savefig("learning_curve_CNN.png")
    plt.close(fig)

    print("Finished training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument("--train_image_path", type=str, default='dataset_combined/faulty')
    parser.add_argument("--train_mask_path", type=str, default='dataset_combined/mask')
    parser.add_argument("--train_pick_dot_path", type=str, default='dataset_combined/pick_dot')
    parser.add_argument("--train_place_dot_path", type=str, default='dataset_combined/place_dot')
    parser.add_argument("--train_csv_path", type=str, default='dataset_combined/failure_label.csv')

    parser.add_argument("--test_image_path", type=str, default='dataset_combined_test/faulty')
    parser.add_argument("--test_mask_path", type=str, default='dataset_combined_test/mask')
    parser.add_argument("--test_pick_dot_path", type=str, default='dataset_combined_test/pick_dot')
    parser.add_argument("--test_place_dot_path", type=str, default='dataset_combined_test/place_dot')
    parser.add_argument("--test_csv_path", type=str, default='dataset_combined_test/failure_label.csv')

    # training parameters
    parser.add_argument("--seed", type=int, default=230)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.00005)
    parser.add_argument("--weight_decay", type=float, default=0.00015)
    args = parser.parse_args()

    main(args)

