#!/bin/bash
script_dir=$(cd $(dirname $0); pwd)


cd weights
# # Trained in https://github.com/Hibiki1020/attitude_estimator_in_timesformer.git
# wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1r4x6xUIx5PeyfdTYnIGVeIljY6UKtJRE' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1r4x6xUIx5PeyfdTYnIGVeIljY6UKtJRE" -O $script_dir/weights/weights_2022_0831_timesformer.pth && rm -rf /tmp/cookies.txt
# wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1LFw0Bj_CxyQBHDh-awE5r3Tfv2VDSQEz' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1LFw0Bj_CxyQBHDh-awE5r3Tfv2VDSQEz" -O $script_dir/weights/weights_2022_1123_timesformer.pth && rm -rf /tmp/cookies.txt

# # https://drive.google.com/file/d/12wG1-Mn-a-O5_mgx3-V3qissycmVx0Lx/view?usp=sharing
# wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=12wG1-Mn-a-O5_mgx3-V3qissycmVx0Lx' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=12wG1-Mn-a-O5_mgx3-V3qissycmVx0Lx" -O $script_dir/weights/weights_2022_12_08_resnet50_pretrain.pth && rm -rf /tmp/cookies.txt


# https://drive.google.com/file/d/1lVgqLxjPCYGZoJ7Fxt5rxm3JteKaah9j/view?usp=sharing
# Pretrain in 8 frame
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1lVgqLxjPCYGZoJ7Fxt5rxm3JteKaah9j' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1lVgqLxjPCYGZoJ7Fxt5rxm3JteKaah9j" -O $script_dir/weights/weights_2022_12_12_resnet50_pretrain_10_epoch_8frame.pth && rm -rf /tmp/cookies.txt


# https://drive.google.com/file/d/1SOM5bwGM_qeGkQ2qO-BEVuvqwlJdZpxQ/view?usp=sharing
# Finetuned in Gimbal, 8 frame
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1SOM5bwGM_qeGkQ2qO-BEVuvqwlJdZpxQ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1SOM5bwGM_qeGkQ2qO-BEVuvqwlJdZpxQ" -O $script_dir/weights/weights_2022_12_22_resnet50_finetune_8_frame.pth && rm -rf /tmp/cookies.txt

# https://drive.google.com/file/d/16ee_aLQ2G3jNEob3D8p56gES4WvYABIU/view?usp=sharing
# Pretrain in 4 frame
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=16ee_aLQ2G3jNEob3D8p56gES4WvYABIU' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=16ee_aLQ2G3jNEob3D8p56gES4WvYABIU" -O $script_dir/weights/weights_2022_12_22_resnet50_pretrain_10_epoch_4frame.pth && rm -rf /tmp/cookies.txt

# https://drive.google.com/file/d/173BrgH4zMfmgSWLkaqnO0ehnUETHBQN_/view?usp=sharing
# Fine tune in 4 frame
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=173BrgH4zMfmgSWLkaqnO0ehnUETHBQN_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=173BrgH4zMfmgSWLkaqnO0ehnUETHBQN_" -O $script_dir/weights/weights_2022_12_22_resnet50_finetune_4frame.pth && rm -rf /tmp/cookies.txt

# https://drive.google.com/file/d/12IkYSc92O4Fn8x_tB913Au9ihLOGmy5H/view?usp=sharing
# Pretrain in 2 frame
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=12IkYSc92O4Fn8x_tB913Au9ihLOGmy5H' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=12IkYSc92O4Fn8x_tB913Au9ihLOGmy5H" -O $script_dir/weights/weights_2022_12_26_resnet50_pretrain_10_epoch_2frame.pth && rm -rf /tmp/cookies.txt

# https://drive.google.com/file/d/1O_cFrGKRBL5k9fFyQ7RSKQCZG16nEmXC/view?usp=sharing
# Pretrain in 1 frame
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1O_cFrGKRBL5k9fFyQ7RSKQCZG16nEmXC' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1O_cFrGKRBL5k9fFyQ7RSKQCZG16nEmXC" -O $script_dir/weights/weights_2022_12_24_resnet50_pretrain_10_epoch_1frame.pth && rm -rf /tmp/cookies.txt

# https://drive.google.com/file/d/1LS786XUI6a7TKc53Mam9SvTren_1gY8H/view?usp=sharing
# Fine tune in 1 frame
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1LS786XUI6a7TKc53Mam9SvTren_1gY8H' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1LS786XUI6a7TKc53Mam9SvTren_1gY8H" -O $script_dir/weights/weights_2022_12_25_resnet50_finetune_1frame.pth && rm -rf /tmp/cookies.txt

# Original Weights of TimeSformer
wget -O TimeSformer_divST_8x32_224_K600.pyth "https://www.dropbox.com/s/4h2qt41m2z3aqrb/TimeSformer_divST_8x32_224_K600.pyth?dl=1"

# Pretrained Weights of TimeSformer in 4 frame(Pretrain)
# https://drive.google.com/file/d/1AY1D7b8rU3YZwRA2br4zb-9oLHPoc0n6/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1AY1D7b8rU3YZwRA2br4zb-9oLHPoc0n6' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1AY1D7b8rU3YZwRA2br4zb-9oLHPoc0n6" -O $script_dir/weights/weights_2023_01_03_TimeSformer_pretrain_4frame.pth && rm -rf /tmp/cookies.txt

# Training Weights of TimeSformer in 4 frame(Fine tune)
# https://drive.google.com/file/d/1arxPxLbsfZaECav_31AsLXvAckAj2rc2/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1arxPxLbsfZaECav_31AsLXvAckAj2rc2' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1arxPxLbsfZaECav_31AsLXvAckAj2rc2" -O $script_dir/weights/weights_2023_01_04_TimeSformer_finetune_4frame.pth && rm -rf /tmp/cookies.txt

# pretrain weights of TimeSformer in 1 frame
# https://drive.google.com/file/d/1dsyu0Pkca0hcDwKub4cNcuFinkm02VmG/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1dsyu0Pkca0hcDwKub4cNcuFinkm02VmG' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1dsyu0Pkca0hcDwKub4cNcuFinkm02VmG" -O $script_dir/weights/weights_2023_01_05_TimeSformer_pretrain_1frame.pth && rm -rf /tmp/cookies.txt

# pretrain weights of TimeSformer in 1 frame (8 depth)
# https://drive.google.com/file/d/1wqFvmWmAiw2rtykAk2Esdz5_Uta7EbFy/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1wqFvmWmAiw2rtykAk2Esdz5_Uta7EbFy' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wqFvmWmAiw2rtykAk2Esdz5_Uta7EbFy" -O $script_dir/weights/weights_2023_01_05_TimeSformer_pretrain_1frame_8depth.pth && rm -rf /tmp/cookies.txt

# Pretrained Weights of TimeSformer in 8 frame(Pretrain)
# https://drive.google.com/file/d/1dZcj_9ZEjhAzKwpy_WvnTiKcoPw6YsRY/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1dZcj_9ZEjhAzKwpy_WvnTiKcoPw6YsRY' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1dZcj_9ZEjhAzKwpy_WvnTiKcoPw6YsRY" -O $script_dir/weights/weights_2023_01_07_TimeSformer_pretrain_8frame.pth && rm -rf /tmp/cookies.txt


# Pretrained Weights of TimeSformer in 8 frame for IROS
# https://drive.google.com/file/d/1a_RTntmXjGk5TJ58pWVo2CCL559jvkcM/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1a_RTntmXjGk5TJ58pWVo2CCL559jvkcM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1a_RTntmXjGk5TJ58pWVo2CCL559jvkcM" -O $script_dir/weights/weights_2023_02_02_TimeSformer_pretrain_8frame_for_IROS.pth && rm -rf /tmp/cookies.txt


# wget https://download.pytorch.org/models/resnet18-f37072fd.pth
# wget https://download.pytorch.org/models/resnet34-b627a593.pth
wget https://download.pytorch.org/models/resnet50-0676ba61.pth
# wget https://download.pytorch.org/models/resnet101-63fe2227.pth
# wget https://download.pytorch.org/models/resnet152-394f9c45.pth

cd $script_dir