import pandas as pd  # type: ignore
import torch
import numpy as np
import torchvision  # type: ignore

from train import set_task
from task_selection import (
    compute_importance_train,
    get_prototypes,
    select_subnetwork,
    select_subnetwork_maxoutput,
    select_subnetwork_free_energy,
    select_subnetwork_icarl,
)


def eval(
    args,
    model,
    train_dataset,
    test_dataset,
    path_to_save,
    device,
    method="IS",
    batch_size=32,
    max_num_learned=10,
    shuffle=False,
    apply_calib=False,
):
    columns = ["seed", "batch_size", "task", "task_acc", "num_samples", "acc"]
    data_GNu = pd.DataFrame(columns=columns)

    model.load_state_dict(torch.load(path_to_save, map_location=device))
    model.eval()

    if method == "IS":
        importances_train, total_importances_train = compute_importance_train(
            model, train_dataset, device
        )
    elif method == "nme":
        prototypes, _ = get_prototypes(model)

    total_acc = []
    total_acc_matrix = np.zeros((max_num_learned, max_num_learned))
    task_acc_matrix = np.zeros((max_num_learned, max_num_learned))

    test_loaders = []

    for i in range(max_num_learned):
        if shuffle:
            order = np.random.permutation(np.arange(len(test_dataset[i])))
            test_dataset[i].data = test_dataset[i].data[order]
            test_dataset[i].targets = test_dataset[i].targets[order]

        test_loaders.append(
            torch.utils.data.DataLoader(
                test_dataset[i], batch_size=batch_size, shuffle=False, num_workers=2
            )
        )

    for num_learned in range(1, max_num_learned + 1):
        total_correct_preds = 0
        total_size = 0
        for task_id in range(0, num_learned):
            dataset_size = len(test_dataset[task_id].data)
            total_size += dataset_size

            acc_task_classification = 0
            correct_preds = 0
            for x, y_true in test_loaders[task_id]:
                x_tmp = torchvision.transforms.RandomHorizontalFlip(p=1)(x)

                x_tmp = torch.cat((x, x_tmp))

                if method == "IS":
                    j0 = select_subnetwork(
                        model, x_tmp, importances_train[:num_learned], device
                    )
                elif "max" in method:
                    j0 = select_subnetwork_maxoutput(
                        model, x_tmp, num_learned, device, apply_calib
                    )
                elif method == "nme":
                    j0 = select_subnetwork_icarl(
                        model, x_tmp, prototypes, num_learned, device
                    )
                elif method == "free_energy":
                    j0 = select_subnetwork_free_energy(
                        model, x_tmp, num_learned, device, apply_calib
                    )

                del x_tmp

                if j0 == task_id:
                    acc_task_classification += x.size(0)

                    set_task(model, j0)

                    pred = model(x.to(device))
                    correct_preds += (
                        (pred.argmax(dim=1) == y_true.to(device)).sum().float()
                    )

            total_correct_preds += correct_preds
            acc_task_classification /= dataset_size

            task_acc_matrix[task_id, num_learned - 1] = acc_task_classification
            total_acc_matrix[task_id, num_learned - 1] = correct_preds / dataset_size

        total_acc.append(100 * (total_correct_preds / total_size).cpu())

        print(
            f"Accuracy after task {task_id+1}: {100*(total_correct_preds/total_size).item():.2f}%"
        )

    acc = total_correct_preds / total_size
    print(f"Accuracy : {100*acc.item():.2f}%")

    return total_acc, task_acc_matrix, total_acc_matrix


def backward_transfer(acc_matrix, tasks_learned):
    bwt = np.zeros(tasks_learned)

    for t in range(1, tasks_learned):
        for task_id in range(t):
            bwt[t] += 100 * (acc_matrix[task_id, t] - acc_matrix[task_id, task_id]) / t

    return list(-np.round(bwt, 2))
