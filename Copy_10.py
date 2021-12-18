import os
from shutil import copyfile

# путь к папке где хранятся файлы
path_src = "f:\Work area\\NN\JPG\\"

# путь к папке куда сохранить файлы
path_dst = "f:\Work area\\NN\Data\\"

os.chdir(path_src)
list_src = os.listdir(path=".")
list_dst = []
for i, j in enumerate(list_src):
    if i%10 == 0:
        list_dst.append(j)

trein_dir = path_dst+'train\\'
test_dir = path_dst+'test\\'
try:
    os.mkdir(path_dst)
except:
    pass
try:
    os.mkdir(trein_dir)
except:
    pass
try:
    os.mkdir(test_dir)
except:
    pass
list_trein = []
list_test = []
for t, i in enumerate(list_dst):
    if t != 0 and t%5 == 0:
        copyfile(path_src+ i, test_dir + i)
        list_test.append(i)
    else:
        copyfile(path_src+i, trein_dir + i)
        list_trein.append(i)
with open(trein_dir+'trein_names.txt', 'w+') as trein_names:
    for i, t in enumerate(list_trein):
        new_name = str(i).zfill(len(t[:-4]))+t[-4:]
        trein_names.write(f'{new_name} = {t}\n')
        os.rename(trein_dir+t, trein_dir+new_name)

with open(test_dir+'test_names.txt', 'w+') as test_names:
    for i, t in enumerate(list_test):
        new_name = str(i).zfill(len(t[:-4])) + t[-4:]
        test_names.write(f'{new_name} = {t}\n')
        os.rename(test_dir + t, test_dir + new_name)