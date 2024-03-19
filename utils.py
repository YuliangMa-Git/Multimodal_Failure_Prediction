import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from sklearn.metrics import confusion_matrix
import seaborn as sns


def loss_fn(pred, y):
    loss = torch.nn.functional.binary_cross_entropy(pred, y)
    return loss


def model_evaluation(data_loader, model, device):
    gt_list = []
    preds_list = []
    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for (image, mask, pick_dot, place_dot, label) in data_loader:
            image = image.to(device)
            mask = mask.to(device)
            pick_dot = pick_dot.to(device)
            place_dot = place_dot.to(device)
            label = label.to(device)
            pred = model(image, mask, pick_dot, place_dot)
            elapsed_time = start_event.elapsed_time(end_event)
            # pred = model(image, pick_dot, place_dot) #Up to the chosen model
            # pred = model(image, mask)
            # pred = model(image)

            gt_list.extend(list(label.cpu().flatten()))
            preds_list.extend(list(pred.cpu().numpy().flatten()))

            loss = loss_fn(pred, label)
            test_loss += loss.item()

        test_loss /= len(data_loader)
    
    return test_loss


def get_F1_measure(data_loader, model, device, threshold):
    gt_list = []
    preds_list = []

    model.eval()

    with torch.no_grad():
        for (image, mask, pick_dot, place_dot, label) in data_loader:
            image = image.to(device)
            mask = mask.to(device)
            pick_dot = pick_dot.to(device)
            place_dot = place_dot.to(device)

            pred = model(image, mask, pick_dot, place_dot)
            # pred = model(image, pick_dot, place_dot) #Up to the chosen model
            # pred = model(image, mask)
            # pred = model(image)
            pred_label = pred > threshold
            gt_list.extend(list(label.flatten()))
            preds_list.extend(list(pred_label.cpu().numpy().flatten()))

    tn, fp, fn, tp = confusion_matrix(gt_list, preds_list).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = (2 * recall * precision) / (recall + precision)

    print("precision is: ", precision)
    print("recall is: ", recall)
    print("f1 score is: ", f1_score)

    cm = confusion_matrix(gt_list, preds_list)
    class_labels = ['Okay', 'Failure']

    # Create a confusion matrix plot
    plt.figure(figsize=(24, 18))
    csfont = {'fontname': 'Times New Roman'}
    sns.set(font_scale=12, font="Times New Roman")
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g',
                xticklabels=class_labels, yticklabels=class_labels, cbar=False, vmin=0, vmax=350)
    plt.savefig('confusion_matrix_CNN.png')  # Save the plot as a PNG file
    plt.show()
    return
