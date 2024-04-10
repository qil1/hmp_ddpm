python train/train_h36.py --save_dir_name 'h36/add_tpred50_dropout0' --t_pred 50 --fusion_add true --gpu_index 1
python test/test_h36.py --save_dir_name 'h36/add_tpred50_dropout0' --t_pred 50 --fusion_add true --gpu_index 1
python train/train_h36_gsps.py --t_his 25 --t_pred 100 --gpu_index 7 --joint_num 17 --S_model_dims 2048 --T_enc_hiddims 4096 --T_dec_hiddims 4096 --save_dir_name 'tmp'

# 3dpw
python train/train_3dpw.py --save_dir_name '3dpw/add_tpred30_dropout0' --t_pred 30 --fusion_add true --gpu_index 4 --data_dir '/home/data/xuanqi/HMPdataset/3DPW/sequenceFiles' --joint_num 23
python train/train_3dpw.py --save_dir_name 'tmp2' --t_pred 30 --data_dir '/home/data/xuanqi/HMPdataset/3DPW/sequenceFiles' --joint_num 23 --gpu_index 5 --S_model_dims 512 --T_enc_hiddims 1024 --T_dec_hiddims 1024

# amass
python train/train_amass.py --save_dir_name 'amass/add_tpred25_dropout0' --t_pred 25 --fusion_add true --gpu_index 5 --data_dir '/home/data/xuanqi/HMPdataset/amass' --joint_num 18
python test/test_amass.py --save_dir_name 'amass/add_tpred25_dropout0' --t_pred 25 --fusion_add true --gpu_index 5 --data_dir '/home/data/xuanqi/HMPdataset/amass' --joint_num 18 --iter 30