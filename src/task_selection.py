import torch
import numpy as np
import torchvision  # type: ignore
from data import mean, std
from train import set_task


def features_out(model, x, device):
    i = 0
    bs = 32
    out = []

    while i + bs < len(x):
        out.append(model.features(x[i : (i + bs)].to(device)).cpu().detach())

        i += bs

    if i < len(x) and i + bs >= len(x):
        out.append(model.features(x[i:].to(device)).cpu().detach())

    out = torch.cat(out)

    return out


def compute_importances(model, signal, device, task_id=0):
    importances = []
    total_importance_per_neuron = []

    fc_weights = model.linear.weight.cpu().detach() * model.tasks_masks[task_id][-1][0]
    # scores = np.abs(signal.mean(dim=0))*fc_weights.abs()
    scores = signal.mean(dim=0) * fc_weights

    total_importance_per_neuron.append(scores.sum(axis=0))

    importances.append(scores)

    importances = torch.cat(importances)

    return importances, total_importance_per_neuron


def joint_importance(model, signal, device, num_learned):
    importances = []
    total_importance_per_neuron = []
    # prototypes_mean, prototypes_std = get_prototypes(model)
    fc_weights = model.linear.weight.cpu().detach()

    for task_id in range(num_learned):
        set_task(model, task_id)
        signal_task = features_out(model, signal, device)

        # scores = torch.abs(signal_task).mean(dim=0)*fc_weights.abs()*model.tasks_masks[task_id][-1][0]
        scores = (
            signal_task.mean(dim=0) * fc_weights * model.tasks_masks[task_id][-1][0]
        )

        total_importance_per_neuron.append(scores.sum(axis=0))

        importances.append(scores)
        del signal_task

    return importances, total_importance_per_neuron


def distance(Test, Train, mask):
    # return torch.sum(torch.abs(Test[mask!=0]-Train[mask!=0])/Train[mask!=0])/mask.sum()
    return torch.sum(torch.abs(Test - Train)) / mask.sum()


def compute_importance_train(model, train_dataset, device):
    importances_train = []
    total_importances_train = []
    for task_id in range(model.num_tasks):
        idx = np.random.permutation(np.arange(len(train_dataset[task_id])))
        x = torch.FloatTensor(train_dataset[task_id].data)[idx]
        x = x.permute(0, 3, 1, 2)
        x = torchvision.transforms.Normalize(mean, std)(x.float() / 255)

        set_task(model, task_id)
        x = features_out(model, x, device)

        importances, total_importance_per_neuron = compute_importances(
            model, x, device, task_id=task_id
        )

        importances_train.append(importances)
        total_importances_train.append(total_importance_per_neuron)

        del importances, total_importance_per_neuron, x

    return importances_train, total_importances_train


def select_subnetwork(model, x, importances_train, device, num_layers=1):
    num_learned = len(importances_train)
    importance_x, total_importance_x = joint_importance(model, x, device, num_learned)
    dists = []
    for j in range(len(importance_x)):
        dist = 0

        for l in range(num_layers):
            dist += distance(
                importance_x[j], importances_train[j], model.tasks_masks[j][-1][0].cpu()
            )

        dists.append(dist.item())

    j0 = np.argmin(dists)

    return j0


def get_prototypes(model):
    prototypes_mean = []
    prototypes_std = []
    for task_id in range(model.num_tasks):
        idx = np.random.permutation(np.arange(len(train_dataset[task_id])))[:2000]
        x = torch.FloatTensor(train_dataset[task_id].data)[idx]
        x = x.permute(0, 3, 1, 2)
        x = torchvision.transforms.Normalize(mean, std)(x.float() / 255)

        set_task(model, task_id)
        x = features_out(model, x)

        prototypes_mean.append(x.mean(dim=0))
        prototypes_std.append(x.std(dim=0))

    return prototypes_mean, prototypes_std


def select_subnetwork_icarl(model, x, prototypes, num_learned=10):
    dists = []
    # prototypes = get_prototypes(model)

    for task_id in range(num_learned):
        set_task(model, task_id)
        out = features_out(model, x)

        dists.append(((out.mean(dim=0) - prototypes[task_id]).abs()).mean())

    j0 = np.argmin(dists)

    return j0


# prototypes_mean, prototypes_std = get_prototypes(net)


def select_subnetwork_maxoutput(model, x, num_learned, device, apply_calib=False):
    max_out = []
    for task_id in range(num_learned):
        set_task(model, task_id)
        preds = model(x.to(device))
        if apply_calib:
            max_out.append(
                (
                    (
                        torch.max(
                            preds[
                                :,
                                task_id
                                * model.num_classes_per_task : (
                                    (task_id + 1) * model.num_classes_per_task
                                ),
                            ],
                            dim=1,
                        )[0]
                        - model.calib_eng_mean[task_id]
                    )
                    / model.calib_eng_std[task_id]
                )
                .sum()
                .cpu()
                .detach()
            )
        else:
            max_out.append(
                torch.max(
                    preds[
                        :,
                        task_id
                        * model.num_classes_per_task : (
                            (task_id + 1) * model.num_classes_per_task
                        ),
                    ],
                    dim=1,
                )[0]
                .sum()
                .cpu()
                .detach()
            )

    j0 = np.argmax(max_out)

    return j0


def select_subnetwork_free_energy(model, x, num_learned, device, apply_calib=False):
    max_out = []
    for task_id in range(num_learned):
        set_task(model, task_id)
        preds = model(x.to(device))
        if apply_calib:
            max_out.append(
                (
                    (
                        torch.logsumexp(
                            preds[
                                :,
                                task_id
                                * model.num_classes_per_task : (
                                    (task_id + 1) * model.num_classes_per_task
                                ),
                            ],
                            dim=1,
                        )
                        - model.calib_mean[task_id]
                    )
                    / model.calib_std[task_id]
                )
                .sum()
                .cpu()
                .detach()
            )
        else:
            max_out.append(
                (
                    torch.logsumexp(
                        preds[
                            :,
                            task_id
                            * model.num_classes_per_task : (
                                (task_id + 1) * model.num_classes_per_task
                            ),
                        ],
                        dim=1,
                    )
                    .sum()
                    .cpu()
                    .detach()
                )
            )

    j0 = np.argmax(max_out)

    return j0
