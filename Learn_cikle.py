from Help_fn.my_model_progress import *
import matplotlib.pyplot as plt
import time


epochs = 1000
batch_size = 3
learning_rate = 1e-3
model_load = True

model = NeuralNetwork().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
spisok_loss_path = 'spisok_loss'
savepath = 'd:\Work Area\Xseg_exstract\weights\model_weights.wgh'

if model_load:
    model.load_state_dict(torch.load(savepath))
    spisok_loss = load_picle_file(spisok_loss_path)
else:
    spisok_loss = [0]
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