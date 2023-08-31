import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import copy
from datetime import datetime

from train import rewrite_parameters, accuracy, set_task


def finetune_train(
    model,
    train_loader_in,
    train_loader_out,
    optimizer,
    old_params,
    m_in,
    m_out,
    task_id,
    device,
):
    model.train()  # enter train mode
    loss_avg = 0.0

    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    # train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))
    for in_set, out_set in zip(train_loader_in, train_loader_out):
        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]

        data, target = data.to(device), target.to(device)

        # forward
        x = model(data)

        # backward
        optimizer.zero_grad()

        offset_a = task_id * model.num_classes_per_task
        offset_b = (task_id + 1) * model.num_classes_per_task

        loss = F.cross_entropy(
            x[: len(in_set[0]), offset_a:offset_b], target - offset_a
        )
        # cross-entropy from softmax distribution to uniform distribution
        Ec_out = -torch.logsumexp(x[len(in_set[0]) :, offset_a:offset_b], dim=1)
        Ec_in = -torch.logsumexp(x[: len(in_set[0]), offset_a:offset_b], dim=1)
        hinge_loss = (
            torch.pow(F.relu(Ec_in - m_in), 2).mean()
            + torch.pow(F.relu(m_out - Ec_out), 2).mean()
        )
        loss += 0.1 * hinge_loss

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            rewrite_parameters(model, old_params, device)

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

    return model, optimizer, loss_avg


def finetune_validate(
    model, valid_loader_in, valid_loader_out, m_in, m_out, task_id, device
):
    model.eval()
    set_task(model, task_id)
    loss_avg = 0.0
    hinge_avg = 0.0

    activation_log_in = []
    activation_log_out = []
    with torch.no_grad():
        for in_set, out_set in zip(valid_loader_in, valid_loader_out):
            data = torch.cat((in_set[0], out_set[0]), 0)
            target = in_set[1]

            data, target = data.to(device), target.to(device)

            # forward
            output = model(data)

            offset_a = task_id * model.num_classes_per_task
            offset_b = (task_id + 1) * model.num_classes_per_task

            loss = F.cross_entropy(
                output[: len(in_set[0]), offset_a:offset_b], target - offset_a
            )

            # cross-entropy from softmax distribution to uniform distribution
            Ec_out = -torch.logsumexp(
                output[len(in_set[0]) :, offset_a:offset_b], dim=1
            )
            Ec_in = -torch.logsumexp(output[: len(in_set[0]), offset_a:offset_b], dim=1)

            hinge_loss = (
                torch.pow(F.relu(Ec_in - m_in), 2).mean()
                + torch.pow(F.relu(m_out - Ec_out), 2).mean()
            )

            activation_log_in.append(Ec_in.detach().cpu().numpy())
            activation_log_out.append(Ec_out.detach().cpu().numpy())

            # test loss average
            loss_avg += float(loss.data)
            hinge_avg += float(hinge_loss.data)

    activation_log_in = np.concatenate(activation_log_in, axis=0)
    activation_log_out = np.concatenate(activation_log_out, axis=0)
    plt.figure(figsize=(8, 6))
    plt.hist(activation_log_in, bins=100, label="energy in", alpha=0.5)
    plt.hist(activation_log_out, bins=100, label="energy out", alpha=0.5)
    plt.legend()
    plt.savefig(f"task_id{task_id}_m_in{m_in}_m_out{m_out}.png")
    return model, loss_avg, hinge_avg


def finetuning_loop(
    model,
    optimizer,
    scheduler,
    train_loader_in,
    train_loader_out,
    valid_loader_in,
    valid_loader_out,
    m_in,
    m_out,
    epochs,
    task_id,
    model_name,
    device,
    file_name="model.pth",
    print_every=1,
):
    best_loss = 1e10
    best_acc = 0
    train_losses = []
    valid_losses = []
    hinge_losses = []

    old_params = copy.deepcopy(model.named_parameters)
    # Train model
    print("TRAINING...")
    for epoch in range(0, epochs):
        # training
        model, optimizer, train_loss = finetune_train(
            model,
            train_loader_in,
            train_loader_out,
            optimizer,
            old_params,
            m_in,
            m_out,
            task_id,
            device,
        )
        train_losses.append(train_loss)

        # validation
        model, valid_loss, hinge_loss = finetune_validate(
            model, valid_loader_in, valid_loader_out, m_in, m_out, task_id, device
        )
        valid_losses.append(valid_loss)
        hinge_losses.append(hinge_loss)
        scheduler.step()

        train_acc = accuracy(model, train_loader_in, device=device)
        valid_acc = accuracy(model, valid_loader_in, device=device)

        if valid_acc > best_acc:
            torch.save(model.state_dict(), file_name)
            best_acc = valid_acc

        if epoch % print_every == (print_every - 1):
            print(
                f"{datetime.now().time().replace(microsecond=0)} --- "
                f"Fine Tune Epoch: {epoch}\t"
                f"Fine Tune Train loss: {train_loss:.4f}\t"
                f"Fine Tune Valid CE loss: {valid_loss:.4f}\t"
                f"Fine Tune Valid Hinge loss: {hinge_loss:.4f}\t"
                f"Fine Tune Train accuracy: {100 * train_acc:.2f}\t"
                f"Fine Tune Valid accuracy: {100 * valid_acc:.2f}"
            )

    return model, (train_losses, valid_losses)


def finetune(
    args,
    model,
    train_loader_in,
    train_loader_out,
    test_loader_in,
    test_loader_out,
    device,
    task_id=0,
):
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.ft_lr,
        momentum=args.ft_momentum,
        weight_decay=args.ft_wd,
        nesterov=True,
    )

    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi)
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.ft_epochs * len(train_loader_in),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / args.ft_lr,
        ),
    )

    model, _ = finetuning_loop(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader_in=train_loader_in,
        train_loader_out=train_loader_out,
        valid_loader_in=test_loader_in,
        valid_loader_out=test_loader_out,
        m_in=args.m_in,
        m_out=args.m_out,
        epochs=args.ft_epochs,
        task_id=task_id,
        model_name=args.model_name,
        device=device,
        file_name=args.path_pretrained_model,
    )

    model.load_state_dict(torch.load(args.path_pretrained_model, map_location=device))

    return model
