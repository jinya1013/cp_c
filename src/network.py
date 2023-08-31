import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import copy


class NonAffineBN(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBN, self).__init__(dim, affine=False)


class AffineBN(nn.BatchNorm2d):
    def __init__(self, dim):
        super(AffineBN, self).__init__(dim, affine=True)


class NonAffineNoStatsBN(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineNoStatsBN, self).__init__(
            dim, affine=False, track_running_stats=False
        )


class MultitaskBN(nn.Module):
    def __init__(self, dim, num_tasks, affine=True):
        super(MultitaskBN, self).__init__()
        if affine:
            self.bns = nn.ModuleList([AffineBN(dim) for _ in range(num_tasks)])
        else:
            self.bns = nn.ModuleList([NonAffineBN(dim) for _ in range(num_tasks)])

    def forward(self, x, task_id):
        return self.bns[task_id](x)


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


def _masks_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.ones_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, args, device, stride=1, option="B"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = MultitaskBN(planes, args.num_tasks)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = MultitaskBN(planes, args.num_tasks)

        self.task_id = 0

        self.shortcut = nn.Sequential()
        self.planes = planes
        self.device = device
        self.in_planes = in_planes
        self.stride = stride

        self.block_masks = self._make_masks(in_planes, planes, stride)

        self.tasks_masks = []

        self.planes = planes
        self.in_planes = in_planes
        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, self.planes // 4, self.planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    MultitaskBN(self.expansion * planes, args.num_tasks),
                )

    def add_mask(self):
        self.tasks_masks.append(copy.deepcopy(self.block_masks))

    def _make_masks(self, in_planes, planes, stride):
        if stride != 1 or in_planes != self.expansion * planes:
            mask = [
                torch.ones(planes, in_planes, 3, 3),
                torch.ones(planes, planes, 3, 3),
                torch.ones(self.expansion * planes, in_planes, 1, 1),
            ]
        else:
            mask = [
                torch.ones(planes, in_planes, 3, 3),
                torch.ones(planes, planes, 3, 3),
                torch.ones(planes),
            ]

        return mask

    def forward(self, x):
        active_conv = self.conv1.weight * self.tasks_masks[self.task_id][0].to(
            self.device
        )
        out = F.conv2d(
            x,
            weight=active_conv,
            bias=None,
            stride=self.conv1.stride,
            padding=self.conv1.padding,
            groups=self.conv1.groups,
        )
        out = F.relu(self.bn1(out, self.task_id))

        active_conv = self.conv2.weight * self.tasks_masks[self.task_id][1].to(
            self.device
        )
        out = F.conv2d(
            out,
            weight=active_conv,
            bias=None,
            stride=self.conv2.stride,
            padding=self.conv2.padding,
            groups=self.conv2.groups,
        )
        out = self.bn2(out, self.task_id)

        if self.stride != 1 or self.in_planes != self.planes:
            shortcut = list(self.shortcut.children())[0]
            active_shortcut = shortcut.weight * self.tasks_masks[self.task_id][-1].to(
                self.device
            )

            bn = list(self.shortcut.children())[1]
            shortcut = F.conv2d(
                x,
                weight=active_shortcut,
                bias=None,
                stride=shortcut.stride,
                padding=shortcut.padding,
                groups=shortcut.groups,
            )
            shortcut = bn(shortcut, self.task_id)
            out += shortcut
        else:
            shortcut = self.shortcut(x)
            out += shortcut * (
                self.tasks_masks[self.task_id][-1]
                .reshape((-1, 1, 1))
                .expand((shortcut.size(-3), shortcut.size(-2), shortcut.size(-1)))
                .to(self.device)
            )

        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, args, device, task_id=0):
        super(ResNet, self).__init__()

        # _outputs = [64, 128, 256, 512]
        _outputs = [21, 42, 85, 170]
        self.in_planes = _outputs[0]

        self.num_blocks = num_blocks

        self.num_classes = args.num_classes
        self.num_classes_per_task = args.num_classes_per_task
        self.num_tasks = 0
        self.args = args

        self.device = device

        self.task_id = task_id

        self.conv1 = nn.Conv2d(
            3, _outputs[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv1_masks = torch.ones(_outputs[0], 3, 3, 3)
        self.bn1 = MultitaskBN(_outputs[0], args.num_tasks)

        self.layer1, self.layer1_masks = self._make_layer(
            block, _outputs[0], num_blocks[0], stride=1
        )
        self.layer2, self.layer2_masks = self._make_layer(
            block, _outputs[1], num_blocks[1], stride=2
        )
        self.layer3, self.layer3_masks = self._make_layer(
            block, _outputs[2], num_blocks[2], stride=2
        )
        self.layer4, self.layer4_masks = self._make_layer(
            block, _outputs[3], num_blocks[3], stride=2
        )

        self.layers_masks = [
            self.layer1_masks,
            self.layer2_masks,
            self.layer3_masks,
            self.layer4_masks,
        ]

        self.linear = nn.Linear(_outputs[3] * block.expansion, self.num_classes)
        self.linear_masks = [
            torch.ones(self.num_classes, _outputs[3] * block.expansion),
            torch.ones(self.num_classes),
        ]

        self.apply(_weights_init)

        self.tasks_masks = []

        self._add_mask(task_id=0)

        self.trainable_mask = copy.deepcopy(self.tasks_masks[0])
        self.masks_union = copy.deepcopy(self.tasks_masks[0])
        self.masks_intersection = copy.deepcopy(self.tasks_masks[0])

        self.calib_mean = nn.Parameter(torch.zeros(100))
        self.calib_std = nn.Parameter(torch.ones(100))

        self.calib_eng_mean = nn.Parameter(torch.zeros(100))
        self.calib_eng_std = nn.Parameter(torch.ones(100))

    def _add_mask(self, task_id):
        self.num_tasks += 1
        network_mask = [
            copy.deepcopy(self.conv1_masks),
            copy.deepcopy(self.layers_masks),
            copy.deepcopy(self.linear_masks),
        ]  # 全部1のマスクをコピー

        self.tasks_masks.append(copy.deepcopy(network_mask))  # masks for task = task_id

        for layer in range(len(network_mask[1])):  # layer x block x 0/1
            for block in range(len(network_mask[1][layer])):
                Block = list(list(self.children())[layer + 2])[block]
                Block.add_mask()
                for conv in range(2):
                    self.tasks_masks[task_id][1][layer][block][
                        conv
                    ] = Block.tasks_masks[task_id][conv]

                self.tasks_masks[task_id][1][layer][block][-1] = Block.tasks_masks[
                    task_id
                ][-1]
        # Set masks on the outputs of other classes zero-------
        index = self.num_classes_per_task * task_id
        if index + self.num_classes_per_task < self.num_classes - 1:
            self.tasks_masks[-1][-1][-1][(index + self.num_classes_per_task) :] = 0
            self.tasks_masks[-1][-1][-2][(index + self.num_classes_per_task) :, :] = 0

        if task_id > 0:
            self.tasks_masks[-1][-1][-1][:index] = 0
            self.tasks_masks[-1][-1][-2][:index, :] = 0
        # -----------------------------------------------------

    def set_masks_union(self, num_learned=-1):
        self.masks_union = copy.deepcopy(self.tasks_masks[0])

        if num_learned < 0:
            num_learned = self.num_tasks

        for id in range(1, num_learned):
            self.masks_union[0] = 1 * torch.logical_or(
                self.masks_union[0], self.tasks_masks[id][0]
            )
            for layer in range(len(self.masks_union[1])):
                for block in range(0, len(self.masks_union[1][layer])):
                    for conv in range(2):
                        self.masks_union[1][layer][block][conv] = 1 * torch.logical_or(
                            self.masks_union[1][layer][block][conv],
                            self.tasks_masks[id][1][layer][block][conv],
                        )

                    self.masks_union[1][layer][block][-1] = 1 * torch.logical_or(
                        self.masks_union[1][layer][block][-1],
                        self.tasks_masks[id][1][layer][block][-1],
                    )

            self.masks_union[-1][0] = 1 * torch.logical_or(
                self.masks_union[-1][0], self.tasks_masks[id][-1][0]
            )
            self.masks_union[-1][1] = 1 * torch.logical_or(
                self.masks_union[-1][1], self.tasks_masks[id][-1][1]
            )

    def set_masks_intersection(self):
        self.masks_intersection = copy.deepcopy(self.tasks_masks[0])

        for id in range(1, self.num_tasks):
            self.masks_intersection[0] = 1 * torch.logical_and(
                self.masks_intersection[0], self.tasks_masks[id][0]
            )
            for layer in range(len(self.masks_intersection[1])):
                for block in range(0, len(self.masks_intersection[1][layer])):
                    for conv in range(2):
                        self.masks_intersection[1][layer][block][
                            conv
                        ] = 1 * torch.logical_and(
                            self.masks_intersection[1][layer][block][conv],
                            self.tasks_masks[id][1][layer][block][conv],
                        )

                    self.masks_intersection[1][layer][block][
                        -1
                    ] = 1 * torch.logical_and(
                        self.masks_intersection[1][layer][block][-1],
                        self.tasks_masks[id][1][layer][block][-1],
                    )

            self.masks_intersection[-1][0] = 1 * torch.logical_and(
                self.masks_intersection[-1][0], self.tasks_masks[id][-1][0]
            )
            self.masks_intersection[-1][1] = 1 * torch.logical_and(
                self.masks_intersection[-1][1], self.tasks_masks[id][-1][1]
            )

    def set_trainable_masks(self, task_id):
        if task_id > 0:
            self.trainable_mask[0] = copy.deepcopy(
                1 * ((self.tasks_masks[task_id][0] - self.masks_union[0]) > 0)
            )  # 現在のタスクの枝刈りにおいてまだ生き残っていてかつ, 過去のタスクで使われていない (self.masks_union = 0) なら, self.trainable_mask = 1 とする
            for layer in range(len(self.trainable_mask[1])):  # layer x block x 0/1
                for block in range(len(self.trainable_mask[1][layer])):
                    Block = list(list(self.children())[layer + 2])[
                        block
                    ]  # self.children 子モジュールをイテレータとして取得
                    for conv in range(2):
                        self.trainable_mask[1][layer][block][conv] = copy.deepcopy(
                            1
                            * (
                                (
                                    self.tasks_masks[task_id][1][layer][block][conv]
                                    - self.masks_union[1][layer][block][conv]
                                )
                                > 0
                            )
                        )

                    if Block.stride != 1 or Block.in_planes != Block.planes:
                        self.trainable_mask[1][layer][block][-1] = copy.deepcopy(
                            1
                            * (
                                (
                                    self.tasks_masks[task_id][1][layer][block][-1]
                                    - self.masks_union[1][layer][block][-1]
                                )
                                > 0
                            )
                        )

            self.trainable_mask[-1][0] = copy.deepcopy(
                1 * ((self.tasks_masks[task_id][-1][0] - self.masks_union[-1][0]) > 0)
            )
            self.trainable_mask[-1][1] = copy.deepcopy(
                1 * ((self.tasks_masks[task_id][-1][1] - self.masks_union[-1][1]) > 0)
            )
        else:
            self.trainable_mask = copy.deepcopy(self.tasks_masks[0])

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        layers_masks = []
        for stride in strides:
            basicblock = block(self.in_planes, planes, self.args, self.device, stride)
            layers.append(basicblock)
            layers_masks.append(basicblock.block_masks)
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers), layers_masks

    def features(self, x):
        active_conv = self.conv1.weight * self.tasks_masks[self.task_id][0].to(
            self.device
        )
        out = F.conv2d(
            x,
            weight=active_conv,
            bias=None,
            stride=self.conv1.stride,
            padding=self.conv1.padding,
        )
        out = F.relu(self.bn1(out, self.task_id))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        active_conv = self.conv1.weight * self.tasks_masks[self.task_id][0].to(
            self.device
        )
        out = F.conv2d(
            x,
            weight=active_conv,
            bias=None,
            stride=self.conv1.stride,
            padding=self.conv1.padding,
        )
        out = F.relu(self.bn1(out, self.task_id))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)

        active_weight = self.linear.weight * self.tasks_masks[self.task_id][-1][0].to(
            self.device
        )
        active_bias = self.linear.bias * self.tasks_masks[self.task_id][-1][1].to(
            self.device
        )
        out = F.linear(out, weight=active_weight, bias=active_bias)

        return out

    def _save_masks(self, file_name="net_masks.pt"):
        masks_database = {}

        for task_id in range(self.num_tasks):
            masks_database["conv1.mask.task{}".format(task_id)] = self.tasks_masks[
                task_id
            ][0]

            for layer in range(len(self.num_blocks)):
                for block in range(self.num_blocks[layer]):
                    for conv_num in range(2):
                        name = "layer{}.{}.conv{}.mask.task{}".format(
                            layer + 1, block, conv_num + 1, task_id
                        )
                        masks_database[name] = self.tasks_masks[task_id][1][layer][
                            block
                        ][conv_num]

                    name = "layer{}.{}.shortcut.mask.task{}".format(
                        layer + 1, block, task_id
                    )
                    masks_database[name] = self.tasks_masks[task_id][1][layer][block][
                        -1
                    ]

            masks_database[
                "linear.weight.mask.task{}".format(task_id)
            ] = self.tasks_masks[task_id][-1][0]
            masks_database[
                "linear.bias.mask.task{}".format(task_id)
            ] = self.tasks_masks[task_id][-1][1]

        torch.save(masks_database, file_name)

    def _load_masks(self, file_name="net_masks.pt", num_tasks=1):
        masks_database = torch.load(file_name)

        for task_id in range(num_tasks):
            self.tasks_masks[task_id][0] = masks_database[
                "conv1.mask.task{}".format(task_id)
            ]

            for layer in range(len(self.num_blocks)):  # layer x block x 0/1
                for block in range(self.num_blocks[layer]):
                    Block = list(list(self.children())[layer + 2])[block]
                    for conv in range(2):
                        name = "layer{}.{}.conv{}.mask.task{}".format(
                            layer + 1, block, conv + 1, task_id
                        )
                        Block.tasks_masks[task_id][conv] = masks_database[name]
                        self.tasks_masks[task_id][1][layer][block][
                            conv
                        ] = Block.tasks_masks[task_id][conv]

                    name = "layer{}.{}.shortcut.mask.task{}".format(
                        layer + 1, block, task_id
                    )
                    Block.tasks_masks[task_id][-1] = masks_database[name]
                    self.tasks_masks[task_id][1][layer][block][-1] = Block.tasks_masks[
                        task_id
                    ][-1]

            self.tasks_masks[task_id][-1][0] = masks_database[
                "linear.weight.mask.task{}".format(task_id)
            ]
            self.tasks_masks[task_id][-1][1] = masks_database[
                "linear.bias.mask.task{}".format(task_id)
            ]

            if task_id + 1 < num_tasks:
                self._add_mask(task_id + 1)

        self.set_masks_union()
        self.set_masks_intersection()


def resnet18(args, device):
    return ResNet(BasicBlock, [2, 2, 2, 2], args, device)


def resnet20(args, device):
    return ResNet(BasicBlock, [3, 3, 3], args, device)


def resnet32(args, device):
    return ResNet(BasicBlock, [5, 5, 5], args, device)


def resnet44(args, device):
    return ResNet(BasicBlock, [7, 7, 7], args, device)


def resnet56(args, device):
    return ResNet(BasicBlock, [9, 9, 9], args, device)


def resnet110(args, device):
    return ResNet(BasicBlock, [18, 18, 18], args, device)


def init_model(args, device):
    if args.model_name == "resnet18":
        model = resnet18(args, device)
    elif args.model_name == "resnet32":
        model = resnet32(args, device)
    model = model.to(device)
    return model


def resnet_total_params(model):
    total_number = 0
    for param_name in list(model.state_dict()):
        param = model.state_dict()[param_name]
        total_number += torch.numel(param[param != 0])

    return total_number


def resnet_total_params_mask(model, task_id=0):
    """
    model.tasks_masks を見て, 1になっているマスクを足し合わせる.
    """
    total_number_conv = 0
    total_number_fc = 0

    for name, param in list(model.named_parameters()):
        if name == "conv1":
            total_number_conv += model.tasks_masks[task_id][0].sum()
        elif "linear" in name:
            if "weight" in name:
                total_number_fc += model.tasks_masks[task_id][-1][0].sum()
            else:
                total_number_fc += model.tasks_masks[task_id][-1][1].sum()
        else:
            for layer in range(len(model.num_blocks)):
                for block in range(model.num_blocks[layer]):
                    if name == "layer{}.{}.conv1.weight".format(layer + 1, block):
                        total_number_conv += model.tasks_masks[task_id][1][layer][
                            block
                        ][0].sum()
                    elif name == "layer{}.{}.conv2.weight".format(layer + 1, block):
                        total_number_conv += model.tasks_masks[task_id][1][layer][
                            block
                        ][1].sum()
                    elif name == "layer{}.{}.shortcut.0.weight".format(
                        layer + 1, block
                    ):
                        total_number_conv += model.tasks_masks[task_id][1][layer][
                            block
                        ][-1].sum()

    total = total_number_conv + total_number_fc

    return total.item(), total_number_conv.item(), total_number_fc.item()


def resnet_total_params_mask_union(model, task_id=0):
    """
    model.mask_union を見て, 1になっているマスクを足し合わせる.
    """
    model.set_masks_union(task_id + 1)
    total_number_conv = 0
    total_number_fc = 0

    for name, param in list(model.named_parameters()):
        if name == "conv1":
            total_number_conv += model.masks_union[0].sum()
        elif "linear" in name:
            if "weight" in name:
                total_number_fc += model.masks_union[-1][0].sum()
            else:
                total_number_fc += model.masks_union[-1][1].sum()
        else:
            for layer in range(len(model.num_blocks)):
                for block in range(model.num_blocks[layer]):
                    if name == "layer{}.{}.conv1.weight".format(layer + 1, block):
                        total_number_conv += model.masks_union[1][layer][block][0].sum()
                    elif name == "layer{}.{}.conv2.weight".format(layer + 1, block):
                        total_number_conv += model.masks_union[1][layer][block][1].sum()
                    elif name == "layer{}.{}.shortcut.0.weight".format(
                        layer + 1, block
                    ):
                        total_number_conv += model.masks_union[1][layer][block][
                            -1
                        ].sum()

    total = total_number_conv + total_number_fc

    return total.item(), total_number_conv.item(), total_number_fc.item()


def resnet_get_architecture(model):
    arch = []
    convs = []
    fc = []

    convs.append(torch.sum(model.conv1_masks.sum(dim=(1, 2, 3)) > 0).item())
    for block, num_block in enumerate(model.num_blocks):
        block_masks = []
        for i in range(num_block):
            block_conv_masks = []
            for conv_num in range(2):
                block_conv_masks.append(
                    torch.sum(
                        model.layers_masks[block][i][conv_num].sum(dim=(1, 2, 3)) > 0
                    ).item()
                )

            block_masks.append(block_conv_masks)

        convs.append(block_masks)

    arch.append(convs)

    fc.append(torch.sum(model.linear_masks[0].sum(dim=0) > 0).item())

    arch.append(fc)

    return arch


def resnet_compute_flops(model):
    arch_conv, arch_fc = resnet_get_architecture(model)
    flops = 0
    k = 3
    h = 32
    w = 32
    in_channels = 3

    flops += (2 * h * w * (in_channels * k**2) - 1) * arch_conv[0]

    in_channels = arch_conv[0]

    for block, num_block in enumerate(model.num_blocks):
        for i in range(num_block):
            for conv_num in range(2):
                if conv_num == 1:
                    out_channels = torch.sum(
                        torch.logical_or(
                            model.layers_masks[block][i][1].sum(dim=(1, 2, 3)),
                            model.layers_masks[block][i][-1],
                        )
                        > 0
                    ).item()
                else:
                    out_channels = arch_conv[1 + block][i][conv_num]

                flops += (2 * h * w * (in_channels * k**2) - 1) * out_channels

            in_channels = out_channels

            flops += h * w * model.layers_masks[block][i][-1].sum().item()

        h /= 2
        w /= 2

    flops += (2 * arch_fc[0] - 1) * 10

    return flops
