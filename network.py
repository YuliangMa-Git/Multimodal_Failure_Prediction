import torch
import torch.nn as nn


class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, image_feature):
        image_feature = self.bn1(self.conv1(image_feature))
        image_feature = self.pool1(nn.functional.relu(image_feature))
        image_feature = self.bn2(self.conv2(image_feature))
        image_feature = self.pool2(nn.functional.relu(image_feature))
        image_feature = self.bn3(self.conv3(image_feature))
        image_feature = self.pool3(nn.functional.relu(image_feature))

        return image_feature


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, cnn_feature):
        cnn_feature = self.bn1(self.conv1(cnn_feature))
        cnn_feature = self.pool1(nn.functional.relu(cnn_feature))
        cnn_feature = self.bn2(self.conv2(cnn_feature))
        cnn_feature = self.pool2(nn.functional.relu(cnn_feature))
        cnn_feature = self.bn3(self.conv3(cnn_feature))
        cnn_feature = self.pool3(nn.functional.relu(cnn_feature))

        return cnn_feature


class FPNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_model = ImageModel()
        self.mask_model = CNNModel()
        self.pick_model = CNNModel()
        self.place_model = CNNModel()

        self.img_L1 = nn.Linear(640, 64)
        self.img_A1 = nn.ReLU()

        self.mask_L1 = nn.Linear(640, 64)
        self.mask_A1 = nn.ReLU()

        self.pick_L1 = nn.Linear(640, 64)
        self.pick_A1 = nn.ReLU()

        self.place_L1 = nn.Linear(640, 64)
        self.place_A1 = nn.ReLU()

        self.fc_L_1 = nn.Linear(256, 128)  # (64+64+64, 128)
        self.fc_A_1 = nn.ReLU()

        self.dropout = nn.Dropout(p=0.5)

        self.fc_L_2 = nn.Linear(128, 32)  # (64+64+64, 128)
        self.fc_A_2 = nn.ReLU()

        self.dropout = nn.Dropout(p=0.5)

        self.fc_L_3 = nn.Linear(32, 1)  # 128
        self.fc_A_3 = nn.Sigmoid()

    def forward(self, img, mask, pick_dot, place_dot):
        image_features = self.image_model(img)
        mask_features = self.mask_model(mask)
        pick_features = self.pick_model(pick_dot)
        place_features = self.place_model(place_dot)

        image_features = torch.flatten(image_features, 1)
        mask_features = torch.flatten(mask_features, 1)
        pick_features = torch.flatten(pick_features, 1)
        place_features = torch.flatten(place_features, 1)

        image_features = self.img_A1(self.img_L1(image_features))
        mask_features = self.mask_A1(self.mask_L1(mask_features))
        pick_features = self.pick_A1(self.pick_L1(pick_features))
        place_features = self.place_A1(self.place_L1(place_features))

        image_features = image_features.unsqueeze(0)
        mask_features = mask_features.unsqueeze(0)
        pick_features = pick_features.unsqueeze(0)
        place_features = place_features.unsqueeze(0)

        cat_features = torch.cat((image_features, mask_features, pick_features, place_features), dim=0)
        cat_features = cat_features.permute(1, 0, 2)
        cat_features = torch.flatten(cat_features, 1)
        cat_features = self.fc_A_1(self.fc_L_1(cat_features))
        cat_features = self.dropout(cat_features)
        cat_features = self.fc_A_2(self.fc_L_2(cat_features))
        cat_features = self.dropout(cat_features)
        pred = self.fc_A_3(self.fc_L_3(cat_features))

        return pred


class FPNN_wo_pp(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_model = ImageModel()
        self.mask_model = CNNModel()

        self.img_L1 = nn.Linear(640, 64)
        self.img_A1 = nn.ReLU()

        self.mask_L1 = nn.Linear(640, 64)
        self.mask_A1 = nn.ReLU()

        self.fc_L_1 = nn.Linear(128, 128)  # (64+64+64, 128)
        self.fc_A_1 = nn.ReLU()

        self.dropout = nn.Dropout(p=0.5)

        self.fc_L_2 = nn.Linear(128, 32)  # (64+64+64, 128)
        self.fc_A_2 = nn.ReLU()

        self.dropout = nn.Dropout(p=0.5)

        self.fc_L_3 = nn.Linear(32, 1)  # 128
        self.fc_A_3 = nn.Sigmoid()

    def forward(self, img, mask):
        image_features = self.image_model(img)
        mask_features = self.mask_model(mask)

        image_features = torch.flatten(image_features, 1)
        mask_features = torch.flatten(mask_features, 1)

        image_features = self.img_A1(self.img_L1(image_features))
        mask_features = self.mask_A1(self.mask_L1(mask_features))

        image_features = image_features.unsqueeze(0)
        mask_features = mask_features.unsqueeze(0)

        cat_features = torch.cat((image_features, mask_features), dim=0)
        cat_features = cat_features.permute(1, 0, 2)
        cat_features = torch.flatten(cat_features, 1)

        cat_features = self.fc_A_1(self.fc_L_1(cat_features))
        cat_features = self.dropout(cat_features)
        cat_features = self.fc_A_2(self.fc_L_2(cat_features))
        cat_features = self.dropout(cat_features)
        pred = self.fc_A_3(self.fc_L_3(cat_features))

        return pred


class FPNN_imgonly(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_model = ImageModel()

        self.img_L1 = nn.Linear(640, 64)
        self.img_A1 = nn.ReLU()

        self.fc_L_1 = nn.Linear(64, 128)  # (64+64+64, 128)
        self.fc_A_1 = nn.ReLU()

        self.dropout = nn.Dropout(p=0.5)

        self.fc_L_2 = nn.Linear(128, 32)  # (64+64+64, 128)
        self.fc_A_2 = nn.ReLU()

        self.dropout = nn.Dropout(p=0.5)

        self.fc_L_3 = nn.Linear(32, 1)  # 128
        self.fc_A_3 = nn.Sigmoid()

    def forward(self, img):
        image_features = self.image_model(img)

        image_features = torch.flatten(image_features, 1)

        image_features = self.img_A1(self.img_L1(image_features))

        image_features = image_features.unsqueeze(0)

        cat_features = image_features
        cat_features = cat_features.permute(1, 0, 2)
        cat_features = torch.flatten(cat_features, 1)

        cat_features = self.fc_A_1(self.fc_L_1(cat_features))
        cat_features = self.dropout(cat_features)
        cat_features = self.fc_A_2(self.fc_L_2(cat_features))
        cat_features = self.dropout(cat_features)
        pred = self.fc_A_3(self.fc_L_3(cat_features))

        return pred


class FPNN_wo_mask(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_model = ImageModel()
        self.pick_model = CNNModel()
        self.place_model = CNNModel()

        self.img_L1 = nn.Linear(640, 64)
        self.img_A1 = nn.ReLU()

        self.pick_L1 = nn.Linear(640, 64)
        self.pick_A1 = nn.ReLU()

        self.place_L1 = nn.Linear(640, 64)
        self.place_A1 = nn.ReLU()

        self.fc_L_1 = nn.Linear(192, 128)  # (64+64+64, 128)
        self.fc_A_1 = nn.ReLU()

        self.dropout = nn.Dropout(p=0.5)

        self.fc_L_2 = nn.Linear(128, 32)  # (64+64+64, 128)
        self.fc_A_2 = nn.ReLU()

        self.dropout = nn.Dropout(p=0.5)

        self.fc_L_3 = nn.Linear(32, 1)  # 128
        self.fc_A_3 = nn.Sigmoid()

    def forward(self, img, pick_dot, place_dot):
        image_features = self.image_model(img)
        pick_features = self.pick_model(pick_dot)
        place_features = self.place_model(place_dot)

        image_features = torch.flatten(image_features, 1)
        pick_features = torch.flatten(pick_features, 1)
        place_features = torch.flatten(place_features, 1)

        image_features = self.img_A1(self.img_L1(image_features))
        pick_features = self.pick_A1(self.pick_L1(pick_features))
        place_features = self.place_A1(self.place_L1(place_features))

        image_features = image_features.unsqueeze(0)
        pick_features = pick_features.unsqueeze(0)
        place_features = place_features.unsqueeze(0)

        cat_features = torch.cat((image_features, pick_features, place_features), dim=0)
        cat_features = cat_features.permute(1, 0, 2)
        cat_features = torch.flatten(cat_features, 1)
        cat_features = self.fc_A_1(self.fc_L_1(cat_features))
        cat_features = self.dropout(cat_features)
        cat_features = self.fc_A_2(self.fc_L_2(cat_features))
        cat_features = self.dropout(cat_features)
        pred = self.fc_A_3(self.fc_L_3(cat_features))

        return pred