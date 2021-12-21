import DFLIMG
from Help_fn.torch_model import *

batch_size = 1
epochs = 1
learning_rate = 1e-3
model_load = True

file_list = list_marked_filse_names(dir_path, True)

data = CustomImageDataset(
    anatation=file_list,
    img_dir=dir_path,
    transform=transform,
    make_predict=True
)

model, input_size, spisok_loss = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
model = model.to(device)
dataloader = DataLoader(data, batch_size=batch_size)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

predict_list = test_loop(dataloader, model, loss_fn, make_predict=True)

for i, pred in enumerate(predict_list):
    save_preduct_in_metadata(dir_path+file_list[0][i], pred)
print('prd write in images')