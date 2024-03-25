python train/train_h36.py --save_dir_name 'add_tpred50_dropout0' --t_pred 50 --fusion_add true --gpu_index 1
python test/test_h36.py --save_dir_name 'add_tpred50_dropout0' --t_pred 50 --fusion_add true --gpu_index 1


# 3dpw
python train/train_3dpw.py --save_dir_name '3dpw/add_tpred30_dropout0' --t_pred 30 --fusion_add true --gpu_index 4 --data_dir '/home/data/xuanqi/HMPdataset/3DPW/sequenceFiles' --joint_num 23


# amass
python train/train_amass.py --save_dir_name 'amass/add_tpred25_dropout0' --t_pred 25 --fusion_add true --gpu_index 5 --data_dir '/home/data/xuanqi/HMPdataset/amass' --joint_num 18
python test/test_amass.py --save_dir_name 'amass/add_tpred25_dropout0' --t_pred 25 --fusion_add true --gpu_index 5 --data_dir '/home/data/xuanqi/HMPdataset/amass' --joint_num 18 --iter 30