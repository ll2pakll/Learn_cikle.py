def train_loop(dataloader, model, loss_fn, optimizer):
    for batch, (X, y, z, k, q) in enumerate(dataloader):
        X, y, z, k, q = X.to(device), y.to(device), z.to(device), k.to(device), q.to(device)
        # Compute prediction and loss
        pr_1, pr_2, pr_3, pr_4 = model(X)
        loss = loss_fn(pr_1, y) + loss_fn(pr_2, z) + loss_fn(pr_3, k) + loss_fn(pr_4, q)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
цацууац