import torch
import argparse

from torch.utils.data import DataLoader
from network import FPNN, FPNN_wo_pp, FPNN_imgonly, FPNN_wo_mask
from dataset_upload import CustomDataset
from utils import model_evaluation, get_F1_measure


def test_all(args):
    test_set = CustomDataset(args.test_image_path, args.test_mask_path, args.test_pick_dot_path,
                             args.test_place_dot_path, args.test_csv_path, 'test')
    test_loader = DataLoader(dataset=test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    fpnn = FPNN().to(device)
    PATH = './CNN_model.pth'
    fpnn.load_state_dict(torch.load(PATH))
    ap = model_evaluation(test_loader, fpnn, device)
    get_F1_measure(test_loader, fpnn, device, 0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--test_image_path", type=str, default='dataset_combined_test/faulty')
    parser.add_argument("--test_mask_path", type=str, default='dataset_combined_test/red_mask')
    parser.add_argument("--test_pick_dot_path", type=str, default='dataset_combined_test/pick_dot')
    parser.add_argument("--test_place_dot_path", type=str, default='dataset_combined_test/place_dot')
    parser.add_argument("--test_csv_path", type=str, default='dataset_combined_test/failure_label.csv')

    args = parser.parse_args()

    # test_datapoint(args)
    test_all(args)
