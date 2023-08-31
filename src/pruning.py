import torch.nn.functional as F
import torch
import numpy as np

from network import resnet_total_params_mask, resnet_total_params
from train import accuracy, training_loop


def resnet_fc_pruning(net, alpha, x_batch, task_id, device, start_fc_prune=0):
    layers = list(net.linear.state_dict())
    num_samples = x_batch.size()[0]

    fc_weight = net.linear.state_dict()[layers[0]] * net.tasks_masks[task_id][-1][0].to(
        device
    )
    fc_bias = net.linear.state_dict()[layers[1]] * net.tasks_masks[task_id][-1][1].to(
        device
    )

    # curr_layer = net.linear(x_batch)
    curr_layer = F.linear(x_batch, weight=fc_weight, bias=fc_bias)

    for i in range(curr_layer.size(1)):
        flow = torch.cat(
            (
                x_batch * fc_weight[i],
                torch.reshape(fc_bias[i].repeat(num_samples), (-1, 1)),
            ),
            dim=1,
        )
        importances = torch.mean(torch.abs(flow), dim=0)

        sum_importance = torch.sum(importances)
        sorted_importances, sorted_indices = torch.sort(importances, descending=True)
        print("sorted_importances: ", sorted_importances)

        cumsum_importances = torch.cumsum(importances[sorted_indices], dim=0)
        pivot = torch.sum(cumsum_importances < alpha * sum_importance)

        if pivot < importances.size(0) - 1:
            pivot += 1
        else:
            pivot = importances.size(0) - 1

        thresh = importances[sorted_indices][pivot]

        net.tasks_masks[task_id][-1][0][i][importances[:-1] <= thresh] = 0

        if importances[-1] <= thresh:
            net.tasks_masks[task_id][-1][1][i] = 0

    # net._apply_mask(task_id)

    return net


def resnet_conv_block_pruning(
    net, layer_num, block_num, conv_num, alpha, x_batch, task_id, residual=0
):
    if layer_num == 0:
        conv = net.conv1
        bn = net.bn1
        active_conv = conv.weight * net.tasks_masks[task_id][0].to(net.device)

        name = "conv1.weight"
        name_bn = "bn1.bns.{}".format(task_id)
    else:
        Block = list(net.named_children())[layer_num + 1][1][block_num]
        if conv_num >= 0:
            conv = list(
                list(net.named_children())[layer_num + 1][1][block_num].named_children()
            )[2 * conv_num][1]
            active_conv = conv.weight * net.tasks_masks[task_id][1][layer_num - 1][
                block_num
            ][conv_num].to(net.device)

            bn = list(
                list(net.named_children())[layer_num + 1][1][block_num].named_children()
            )[2 * conv_num + 1][1]
            name = "layer{}.{}.conv{}.weight".format(layer_num, block_num, conv_num + 1)
            name_bn = "layer{}.{}.bn{}.bns.{}".format(
                layer_num, block_num, conv_num + 1, task_id
            )
        else:
            conv = list(Block.named_children())[-1][1][0]
            active_conv = conv.weight * net.tasks_masks[task_id][1][layer_num - 1][
                block_num
            ][-1].to(net.device)

            bn = list(Block.named_children())[-1][1][1]
            name = "layer{}.{}.shortcut.0.weight".format(layer_num, block_num)
            name_bn = "layer{}.{}.shortcut.1.bns.{}".format(
                layer_num, block_num, task_id
            )

    bn_out = bn(
        F.conv2d(x_batch, weight=active_conv, stride=conv.stride, padding=conv.padding),
        task_id,
    )

    if conv_num == 1:
        block_out = F.relu(bn_out + residual)
    else:
        if conv_num >= 0:
            block_out = F.relu(bn_out)
        else:
            block_out = bn_out

    block_out_mean = block_out.mean(dim=0)

    padding = conv.padding
    stride = conv.stride
    kernel_size = conv.kernel_size
    zero_kernel = torch.zeros(kernel_size)

    filters = net.state_dict()[name]

    p2d = (padding[0],) * 2 + (padding[1],) * 2
    n = x_batch.size(3)
    m = x_batch.size(2)

    x_batch = F.pad(x_batch, p2d, "constant", 0)

    for k in range(filters.size(0)):
        if (block_out_mean[k]).norm(dim=(0, 1)) == 0:
            if layer_num == 0:
                net.tasks_masks[task_id][0][k] = zero_kernel
            else:
                net.tasks_masks[task_id][1][layer_num - 1][block_num][conv_num][
                    k
                ] = zero_kernel
                if conv_num == 1:
                    if (
                        Block.stride != 1
                        or Block.in_planes != Block.expansion * Block.planes
                    ):
                        net.tasks_masks[task_id][1][layer_num - 1][block_num][-1][
                            k
                        ] = torch.zeros((1, 1))

                        shortcut_name_bn = "layer{}.{}.shortcut.1.bns.{}".format(
                            layer_num, block_num, task_id
                        )

                        net.state_dict()[shortcut_name_bn + ".weight"][k] = 0
                        net.state_dict()[shortcut_name_bn + ".bias"][k] = 0
                        net.state_dict()[shortcut_name_bn + ".running_mean"][k] = 0
                        net.state_dict()[shortcut_name_bn + ".running_var"][k] = 0
                    else:
                        net.tasks_masks[task_id][1][layer_num - 1][block_num][-1][k] = 0

            net.state_dict()[name_bn + ".weight"][k] = 0
            net.state_dict()[name_bn + ".bias"][k] = 0
            net.state_dict()[name_bn + ".running_mean"][k] = 0
            net.state_dict()[name_bn + ".running_var"][k] = 0
        else:
            importances = torch.zeros(
                filters.size(1),
                ((n + 2 * padding[0] - kernel_size[0]) // stride[0] + 1),
                ((m + 2 * padding[1] - kernel_size[1]) // stride[1] + 1),
            )

            for i in range(
                kernel_size[0] // 2,
                (n + 2 * padding[0]) - kernel_size[0] // 2,
                stride[0],
            ):
                for j in range(
                    kernel_size[1] // 2,
                    (m + 2 * padding[1]) - kernel_size[1] // 2,
                    stride[1],
                ):
                    input = (
                        x_batch[
                            :,
                            :,
                            (i - kernel_size[0] // 2) : (i + kernel_size[0] // 2 + 1),
                            (j - kernel_size[1] // 2) : (j + kernel_size[1] // 2 + 1),
                        ]
                        .abs()
                        .mean(dim=0)
                    )

                    importances[
                        :,
                        (i - kernel_size[0] // 2) // stride[0],
                        (j - kernel_size[1] // 2) // stride[1],
                    ] = torch.sum(torch.abs(input * filters[k]), dim=(1, 2))

            importances = torch.norm(importances, dim=(1, 2))
            sorted_importances, sorted_indices = torch.sort(
                importances, dim=0, descending=True
            )
            print("sorted_importances: ", sorted_importances)

            pivot = torch.sum(
                sorted_importances.cumsum(dim=0) < alpha * importances.sum()
            )
            if pivot < importances.size(0) - 1:
                pivot += 1
            else:
                pivot = importances.size(0) - 1

            # delete all connectons that are less important than the pivot
            thresh = sorted_importances[pivot]
            kernel_zero_idx = (
                torch.nonzero(importances <= thresh).reshape(1, -1).squeeze(0)
            )

            if layer_num == 0:
                net.tasks_masks[task_id][0][k][kernel_zero_idx] = zero_kernel
            else:
                net.tasks_masks[task_id][1][layer_num - 1][block_num][conv_num][k][
                    kernel_zero_idx
                ] = zero_kernel

    if conv_num == 1:
        pruned_channels = (
            torch.nonzero(
                bn_out.abs().mean(dim=0).norm(dim=(1, 2))
                / residual.abs().mean(dim=0).norm(dim=(1, 2))
                < (1 - alpha) / alpha
            )
            .reshape(1, -1)
            .squeeze(0)
        )
        net.tasks_masks[task_id][1][layer_num - 1][block_num][conv_num][
            pruned_channels
        ] = zero_kernel

        net.state_dict()[name_bn + ".weight"][pruned_channels] = 0
        net.state_dict()[name_bn + ".bias"][pruned_channels] = 0
        net.state_dict()[name_bn + ".running_mean"][pruned_channels] = 0
        net.state_dict()[name_bn + ".running_var"][pruned_channels] = 0
        """
        pruned_channels = torch.nonzero(
            residual.abs().mean(dim=0).norm(dim=(1, 2))/bn_out.abs().mean(dim=0).norm(dim=(1, 2)) < (1-alpha)/alpha).reshape(1, -1).squeeze(0)
        
        if Block.stride != 1 or Block.in_planes != Block.expansion*Block.planes:
            net.tasks_masks[task_id][1][layer_num-1][block_num][-1][pruned_channels] = torch.zeros((1, 1))

            shortcut_name_bn = 'layer{}.{}.shortcut.1.bns.{}'.format(layer_num, block_num, task_id)

            #net.state_dict()[shortcut_name_bn+'.weight'][pruned_channels] = 0
            #net.state_dict()[shortcut_name_bn+'.bias'][pruned_channels] = 0
            net.state_dict()[shortcut_name_bn+'.running_mean'][pruned_channels] = 0
            net.state_dict()[shortcut_name_bn+'.running_var'][pruned_channels] = 0  
        else: 
            net.tasks_masks[task_id][1][layer_num-1][block_num][-1][pruned_channels] = 0
        """
    # net._apply_mask(task_id)
    # print(name)
    return net, block_out


def resnet_conv_pruning(net, alpha, x_batch, start_conv_prune, task_id, device):
    net.eval()
    named_params = list(net.named_parameters())

    for name, param in named_params:
        if name == "conv1.weight":
            net, x_batch = resnet_conv_block_pruning(
                net, 0, 0, 0, alpha, x_batch, task_id
            )
        else:
            for layer in range(len(net.num_blocks)):
                for block in range(net.num_blocks[layer]):
                    Block = list(net.named_children())[layer + 2][1][block]
                    for conv_num in range(2):
                        if name == "layer{}.{}.conv{}.weight".format(
                            layer + 1, block, conv_num + 1
                        ):
                            if conv_num == 0:
                                if (
                                    Block.stride != 1
                                    or Block.in_planes != Block.expansion * Block.planes
                                ):
                                    net, residual = resnet_conv_block_pruning(
                                        net,
                                        layer + 1,
                                        block,
                                        -1,
                                        alpha,
                                        x_batch,
                                        task_id,
                                        residual=0,
                                    )
                                else:
                                    residual = list(Block.named_children())[-1][1](
                                        x_batch
                                    )

                            net, x_batch = resnet_conv_block_pruning(
                                net,
                                layer + 1,
                                block,
                                conv_num,
                                alpha,
                                x_batch,
                                task_id,
                                residual,
                            )
    return net


def resnet_backward_pruning(net, task_id):
    pruned_channels = (
        torch.nonzero(net.tasks_masks[task_id][-1][0].sum(dim=0) == 0)
        .reshape(1, -1)
        .squeeze(0)
    )

    kernel_size = list(list(net.layer3.named_children())[-1][1].named_children())[-3][
        1
    ].kernel_size
    zero_kernel = torch.zeros(kernel_size)

    for name in reversed(list(net.state_dict())):
        for layer in range(len(net.num_blocks))[::-1]:
            for block in range(net.num_blocks[layer])[::-1]:
                Block = list(net.children())[layer + 2][block]

                for conv_num in range(2)[::-1]:
                    if (
                        "layer{}.{}.bn{}.bns.{}".format(
                            layer + 1, block, conv_num + 1, task_id
                        )
                        in name
                    ) and ("num_batches_tracked" not in name):
                        net.state_dict()[name][pruned_channels] = 0
                    elif name == "layer{}.{}.conv{}.weight".format(
                        layer + 1, block, conv_num + 1
                    ):
                        net.tasks_masks[task_id][1][layer][block][conv_num][
                            pruned_channels
                        ] = zero_kernel

                        if conv_num == 1:
                            if (
                                Block.stride != 1
                                or Block.in_planes != Block.expansion * Block.planes
                            ):
                                net.tasks_masks[task_id][1][layer][block][-1][
                                    pruned_channels
                                ] = torch.zeros((1, 1))
                                name_bn = "layer{}.{}.shortcut.1.bns.{}".format(
                                    layer + 1, block, task_id
                                )

                                net.state_dict()[name_bn + ".weight"][
                                    pruned_channels
                                ] = 0
                                net.state_dict()[name_bn + ".bias"][pruned_channels] = 0
                                net.state_dict()[name_bn + ".running_mean"][
                                    pruned_channels
                                ] = 0
                                net.state_dict()[name_bn + ".running_var"][
                                    pruned_channels
                                ] = 0
                            else:
                                net.tasks_masks[task_id][1][layer][block][-1][
                                    pruned_channels
                                ] = 0

                            pruned_channels = (
                                torch.nonzero(
                                    net.tasks_masks[task_id][1][layer][block][
                                        conv_num
                                    ].sum(dim=(0, 2, 3))
                                    == 0
                                )
                                .reshape(1, -1)
                                .squeeze(0)
                            )
                        else:
                            if (
                                Block.stride != 1
                                or Block.in_planes != Block.expansion * Block.planes
                            ):
                                pruned_channels = (
                                    torch.nonzero(
                                        (
                                            net.tasks_masks[task_id][1][layer][block][
                                                -1
                                            ].sum(dim=(0, 2, 3))
                                            == 0
                                        )
                                        * (
                                            net.tasks_masks[task_id][1][layer][block][
                                                conv_num
                                            ].sum(dim=(0, 2, 3))
                                            == 0
                                        )
                                    )
                                    .reshape(1, -1)
                                    .squeeze(0)
                                )
                            else:
                                pruned_channels = (
                                    torch.nonzero(
                                        (
                                            1
                                            - net.tasks_masks[task_id][1][layer][block][
                                                -1
                                            ]
                                        )
                                        * (
                                            net.tasks_masks[task_id][1][layer][block][
                                                conv_num
                                            ].sum(dim=(0, 2, 3))
                                            == 0
                                        )
                                    )
                                    .reshape(1, -1)
                                    .squeeze(0)
                                )

    net.tasks_masks[task_id][0][pruned_channels] = zero_kernel
    net.state_dict()["bn1.bns.{}.weight".format(task_id)][pruned_channels] = 0
    net.state_dict()["bn1.bns.{}.bias".format(task_id)][pruned_channels] = 0
    net.state_dict()["bn1.bns.{}".format(task_id) + ".running_mean"][
        pruned_channels
    ] = 0
    net.state_dict()["bn1.bns.{}".format(task_id) + ".running_var"][pruned_channels] = 0

    return net


def resnet_pruning(
    net, alpha_conv, x_batch, task_id, device, start_conv_prune=0, start_fc_prune=0
):
    # do forward step fon convolutional layers
    # should be replaced by conv layers pruning and then the forward step
    if start_conv_prune >= 0:
        net = resnet_conv_pruning(
            net, alpha_conv, x_batch, start_conv_prune, task_id, device
        )

    x_batch = net.features(x_batch)

    print("---Before backward: ", resnet_total_params_mask(net, task_id))
    net = resnet_backward_pruning(net, task_id)
    print("---After backward: ", resnet_total_params_mask(net, task_id))

    # net._apply_mask(task_id)

    return net


def iterative_pruning(
    args,
    net,
    train_loader,
    test_loader,
    x_prune,
    task_id,
    device,
    path_to_save,
    start_conv_prune=0,
    start_fc_prune=-1,
):
    cr = 1
    sparsity = 100
    acc = np.round(100 * accuracy(net, test_loader, device), 2)

    init_masks_num = resnet_total_params_mask(net, task_id)

    for it in range(1, args.num_iters + 1):
        # before_params_num = total_params(net)
        before_masks_num = resnet_total_params_mask(
            net, task_id
        )  # tasks_masks の 1 をカウントする
        net.eval()
        print(
            "The percentage of the remaining weights: ",
            100 * before_masks_num[0] / resnet_total_params(net),
        )
        net = resnet_pruning(
            net=net,
            alpha_conv=args.alpha_conv,
            x_batch=x_prune,
            task_id=task_id,
            device=device,
            start_conv_prune=start_conv_prune,
        )

        net.set_trainable_masks(task_id)

        after_masks_num = resnet_total_params_mask(net, task_id)
        acc_before = np.round(100 * accuracy(net, test_loader, device), 2)
        # curr_arch = lenet_get_architecture(net)
        print("Accuracy before retraining: ", acc_before)
        print(
            "Compression rate on iteration %i: " % it,
            before_masks_num[0] / after_masks_num[0],
        )
        print("Total compression rate: ", init_masks_num[0] / after_masks_num[0])
        print(
            "The percentage of the remaining weights: ",
            100 * after_masks_num[0] / resnet_total_params(net),
        )
        # print('Architecture: ', curr_arch)

        cr = np.round(init_masks_num[0] / after_masks_num[0], 2)
        sparsity = np.round(100 * after_masks_num[0] / init_masks_num[0], 2)

        if args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                net.parameters(), lr=args.lr, weight_decay=args.wd
            )
        elif args.optimizer == "radam":
            optimizer = RAdam(
                net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd
            )
        else:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd
            )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.decay_epochs_retrain, gamma=args.gamma
        )
        net, _ = training_loop(
            model=net,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            valid_loader=test_loader,
            epochs=args.retrain_epochs,
            task_id=task_id,
            model_name=args.model_name,
            device=device,
            file_name=path_to_save,
        )

        net.load_state_dict(torch.load(path_to_save, map_location=device))

        acc_after = np.round(100 * accuracy(net, test_loader, device), 2)
        print("Accuracy after retraining: ", acc_after)

        print("-------------------------------------------------")

    return net
