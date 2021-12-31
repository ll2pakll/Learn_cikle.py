from Help_fn.torch_model import *

list_manager = Lists_manager(dir_path)
marked_file_list = list_manager.get_target_list()
file_list = list_manager.get_image_list()
default_marker = np.array([[0, 0]] * 4, np.int32)

inter_points = Inter_points(idx=0,
                            file_list=file_list,
                            dir_path=dir_path,
                            previous_idx=list_manager.get_previous_marked_idx(0),
                            next_idx=list_manager.get_next_marked_idx(0)
                            )

for i, n in enumerate(file_list):
    dflimg = DFLIMG.DFLJPG.load(dir_path + n)
    if i % 50 == 0 and i:
        print(f'{i}/{len(file_list)}')
    if marked_file_list[i][1]:
        markers = dflimg.get_dict()['keypoints']
    else:
        markers = default_marker
    inter_points.set_data(idx=i,
                          previous_idx=list_manager.get_previous_marked_idx(i),
                          next_idx=list_manager.get_next_marked_idx(i),
                          marker=markers)
    meta = dflimg.get_dict()
    try:
        meta['inter_points'] = inter_points.get_points()
    except:
        meta = {'inter_points': inter_points.get_points()}
    dflimg.set_dict(dict_data=meta)
    dflimg.save()