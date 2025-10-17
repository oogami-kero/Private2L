import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class DropBlock(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def forward(self, x, gamma):
        if not self.training:
            return x
        b, c, h, w = x.shape
        mask = torch.bernoulli(torch.full((b, c, h - (self.block_size - 1), w - (self.block_size - 1)), gamma,
                                          device=x.device))
        left = int((self.block_size - 1) / 2)
        right = int(self.block_size / 2)
        pad_mask = F.pad(mask, (left, right, left, right))
        non_zero = (mask > 0).nonzero()
        if non_zero.numel() > 0:
            offsets = torch.stack([
                torch.arange(self.block_size, device=x.device).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1),
                torch.arange(self.block_size, device=x.device).repeat(self.block_size),
            ], dim=1)
            offsets = torch.cat((torch.zeros(self.block_size ** 2, 2, device=x.device).long(), offsets.long()), 1)
            idx = non_zero.repeat(self.block_size ** 2, 1) + offsets.repeat(non_zero.shape[0], 1)
            pad_mask[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]] = 1.0
        block_mask = 1 - pad_mask
        countM = block_mask.numel()
        count_ones = block_mask.sum()
        return block_mask * x * (countM / (count_ones + 1e-12))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        self.num_batches_tracked += 1
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        if self.drop_rate > 0:
            if self.drop_block:
                feat = out.size(2)
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat ** 2 / (feat - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)
        return out


class ResNet(nn.Module):
    def __init__(self, block, keep_prob=1.0, avg_pool=False, drop_rate=0.0, dropblock_size=5):
        super().__init__()
        self.inplanes = 3
        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.keep_avg_pool = avg_pool
        if avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size)]
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.keep_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


def resnet12(keep_prob=1.0, avg_pool=False, drop_rate=0.0, **kwargs):
    return ResNet(BasicBlock, keep_prob=keep_prob, avg_pool=avg_pool, drop_rate=drop_rate, **kwargs)


class P2LImageModel(nn.Module):
    """Image model with a small privacy transform layer + frozen backbone.

    Forward returns (embedding, proj_or_same, logits) to be compatible with F2L training calls.
    - If all_classify=False: classify via few-shot head on embeddings.
    - If all_classify=True: project via MLP and classify over all classes (for contrastive supervision).
    """

    def __init__(self, n_classes: int, total_classes: int, out_dim: int = 256, dataset: str = "FC100", freeze_backbone: bool = True):
        super().__init__()
        # Small transform at input to absorb DP noise and reduce trainable size
        self.privacy_tl = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
        # Lightweight backbone
        if dataset == "FC100":
            self.features = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=2)
        else:
            self.features = resnet12(avg_pool=True, drop_rate=0.1)
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        num_ftrs = 640
        # Projection head (used when all_classify=True)
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

        # Few-shot head
        self.few_classify = nn.Linear(num_ftrs, n_classes)

        # All-class head for auxiliary supervision
        self.all_classify = nn.Linear(out_dim, total_classes)

    def forward(self, x, all_classify: bool = False):
        x = self.privacy_tl(x)
        h = self.features(x)  # [B, C]
        h = h.view(h.size(0), -1)
        if not all_classify:
            y = self.few_classify(h)
            return h, h, y
        else:
            z = self.l1(h)
            z = F.relu(z)
            z = self.l2(z)
            y = self.all_classify(z)
            return h, z, y
