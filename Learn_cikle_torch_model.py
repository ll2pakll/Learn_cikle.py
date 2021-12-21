from Help_fn.torch_model import *
import matplotlib.pyplot as plt
import time


epochs = 1
batch_size = 4
learning_rate = 1e-3

model, input_size, spisok_loss = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
model = model.to(device)

if feature_extract:
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)
else:
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
end_counter = epochs


for t in range(epochs):
    spisok_loss[0] += 1
    print(f"Epoch {spisok_loss[0]}\n{end_counter} left")
    if t != 0 and t % 100 == 0:
        torch.save(model.state_dict(), savepath)
        print(f'Model saved in {savepath}')
        save_picle_file(spisok_loss, spisok_loss_path)
    start_time = time.time()
    train_loop(train_dataloader, model, loss_fn, optimizer, spisok_loss)
    end_counter -= 1
    print(f"sec {(time.time() - start_time):.3f}\n----------------------------")

torch.save(model.state_dict(), savepath)
save_picle_file(spisok_loss, spisok_loss_path)
print(f'Model saved in {savepath}')
print("Done!")

plt.figure(figsize=(40, 40))
plt.plot(spisok_loss[-epochs:])
plt.show()