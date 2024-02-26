import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from sklearn.neighbors import KernelDensity
from sklearn.metrics import average_precision_score, confusion_matrix
import seaborn as sns


# plt.rcParams.update({
#    'font.size': 24
# })

def loss_fn(pred, y):
    loss = torch.nn.functional.binary_cross_entropy(pred, y)
    return loss


def model_evaluation(data_loader, model, device):
    gt_list = []
    preds_list = []
    test_loss = 0.0
    infer_time_list = []
    model.eval()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        for (image, mask, pick_dot, place_dot, label) in data_loader:
            image = image.to(device)
            mask = mask.to(device)
            pick_dot = pick_dot.to(device)
            place_dot = place_dot.to(device)
            label = label.to(device)
            start_event.record()
            # pred = model(image, mask, pick_dot, place_dot)
            end_event.record()

            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            # print(f"Inference time: {elapsed_time} milliseconds")
            # infer_time_list.append(elapsed_time)
            # pred = model(image, mask)
            pred = model(image, pick_dot, place_dot)
            # pred = model(image)

            gt_list.extend(list(label.cpu().flatten()))
            preds_list.extend(list(pred.cpu().numpy().flatten()))

            loss = loss_fn(pred, label)
            test_loss += loss.item()

        test_loss /= len(data_loader)
        # del infer_time_list[0]
        # del infer_time_list[-1]
        # del infer_time_list[-2]
        # print(len(infer_time_list))
        # print("inference time average is :  ", sum(infer_time_list)/len(infer_time_list))
    # return sum(infer_time_list)/len(infer_time_list)
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

            # pred = model(image, mask, pick_dot, place_dot)
            # pred = model(image, mask)
            pred = model(image, pick_dot, place_dot)
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
    # plt.xlabel('Predicted', **csfont)
    # plt.ylabel('Labels', **csfont)
    # plt.title('Confusion Matrix', **csfont)
    plt.savefig('confusion_matrix_CNN.png')  # Save the plot as a PNG file
    plt.show()
    return


def logit(x):
    return np.log(x / (1 - x))


def logistic(x):
    return np.exp(x) / (1 + np.exp(x))


def density_estimation(data_loader, model, device, threshold=None):
    normal_score = []
    failure_score = []

    model.eval()
    with torch.no_grad():
        for (image, mask, pick_dot, place_dot, label) in data_loader:
            image = image.to(device)
            mask = mask.to(device)
            pick_dot = pick_dot.to(device)
            place_dot = place_dot.to(device)

            # pred = model(image, mask, pick_dot, place_dot)
            # pred = model(image, mask)
            pred = model(image, pick_dot, place_dot)
            # pred = model(image)

            label = label.flatten()
            pred = pred.cpu().numpy().flatten()

            for label_i, pred_i in zip(label, pred):
                if label_i == 0:
                    normal_score.append([pred_i])
                else:
                    failure_score.append([pred_i])

    normal_score = np.array(normal_score)
    failure_score = np.array(failure_score)

    normal_score = np.clip(normal_score, a_min=1e-6, a_max=1 - 1e-6)
    failure_score = np.clip(failure_score, a_min=1e-6, a_max=1 - 1e-6)

    # transformation trick
    normal_score_tf, failure_score_tf = logit(normal_score), logit(failure_score)

    kde_normal = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(normal_score_tf)
    kde_failure = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(failure_score_tf)
    X_plot = np.linspace(0.001, 0.999, 1000)[:, np.newaxis]
    X_plot_tf = logit(X_plot)
    log_dens_normal_tf = kde_normal.score_samples(X_plot_tf)
    log_dens_failure_tf = kde_failure.score_samples(X_plot_tf)

    dens_normal = np.exp(log_dens_normal_tf)[:, np.newaxis] / (X_plot * (1 - X_plot))
    dens_failure = np.exp(log_dens_failure_tf)[:, np.newaxis] / (X_plot * (1 - X_plot))

    if not threshold:
        plt.figure(figsize=(10, 9))
        plt.fill_between(X_plot[:, 0], dens_normal[:, 0], fc='#AAAAFF', alpha=0.3, label='normal')
        plt.fill_between(X_plot[:, 0], dens_failure[:, 0], fc='#FFAAAA', alpha=0.3, label='failure')
        plt.xlabel('Score')
        plt.ylabel('Density')
        plt.legend()
        plt.axis([0, 1, 0, 5])
        plt.savefig('FPNN_pdf.png')
        plt.show()
    else:
        plt.figure(figsize=(10, 9))
        plt.fill_between(X_plot[:, 0], dens_normal[:, 0], fc='#0066CC', alpha=0.3)
        plt.fill_between(X_plot[:, 0], dens_failure[:, 0], fc='#FA6C00', alpha=0.3)
        plt.plot(X_plot[:, 0], dens_normal[:, 0], color='#0066CC', linewidth=3.0, label='normal')
        plt.plot(X_plot[:, 0], dens_failure[:, 0], color='#FA6C00', linewidth=3.0, label='failure')
        plt.axvline(x=threshold, ymin=0, ymax=1,
                    color='black', ls='--', linewidth=2.0, label='threshold')
        plt.xlabel('Score')
        plt.ylabel('Density')
        plt.legend(loc='upper right')
        plt.axis([0, 1, 0, 5])
        plt.savefig('FPNN_pdf.png')
        plt.show()
