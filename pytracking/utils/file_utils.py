import os
import shutil
import glob

# VOT_root_dir = r'D:\Tracking\Datasets\VOT2018'

# for dir in os.listdir(VOT_root_dir):
#     if dir[-4:] == "json":
#         continue
#     color_path = os.path.join(VOT_root_dir, dir, 'color')
#     if not os.path.exists(color_path):
#         os.mkdir(color_path)
#     jpg_list = glob.glob(os.path.join(VOT_root_dir, dir, '*.jpg'))
#     for jpg_path in jpg_list:
#         shutil.move(jpg_path, color_path)

# UAV_root_dir = r"D:\Tracking\VisTrack\pytracking\results\tracking_results\fudimp"
# for dir in os.listdir(UAV_root_dir):
#     uav_path = os.path.join(UAV_root_dir, 'UAV123', dir)
#     if not os.path.exists(uav_path):
#         os.makedirs(uav_path)
#         txt_list = glob.glob(os.path.join(UAV_root_dir, dir, 'uav_*.txt'))
#         for txt_path in txt_list:
#             shutil.copy(txt_path, uav_path)

base_path = r'D:\Tracking\VisTrack\pytracking\results\UAV123'
for dir in os.listdir(base_path):
    txt_list = glob.glob(os.path.join(base_path, dir, '*.txt'))
    for f in txt_list:
        # o_path = os.path.join(base_path, f)
        n_name = f.split('\\')[-1][4:]
        n_path = os.path.join(base_path, dir, n_name)
        os.rename(f, n_path)
print('done!')