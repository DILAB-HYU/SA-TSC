

### model saved at 
# "{save dir} \ {model_name} \ {data_set} \ {exp_name} \ {batchsize kernel size} \ spatio_100epoch.pth"


## SleepEDF
# 1. pretrain 
# python main_aug_v2.py --model_name SATSC --exp_name aaai --batch_size 128 --data_set SleepEDF --kernel_size 8 --mu 100.0 --gumbel_tmp 1.0 --aug_mode cross_domain --gumbel_only False --beta 1 --alpha 1

# 2. finetune 
# python main_finetune.py --model_name  SATSC --exp_name aaai --batch_size 128 --data_set SleepEDF  --subject 2 --file_name 1288 --file_epoch 100 --spt_lr 0.0001 --device cuda --epoch 100 




## ISRUC
# 1. pretrain 
# python main_aug_v2.py --model_name SATSC --exp_name aaai --batch_size 128 --data_set ISRUC --kernel_size 8 --mu 100.0 --gumbel_tmp 1.0 --aug_mode cross_domain --gumbel_only False --beta 1 --alpha 1

# 2. finetune 
# python main_finetune.py --model_name  SATSC --exp_name aaai --batch_size 128 --data_set ISRUC  --subject 2 --file_name 1288 --file_epoch 100 --spt_lr 0.001 --device cuda --epoch 100 




## HAR 
# 1. pretrain 
# python main_aug_v2.py --model_name SATSC --exp_name aaai --batch_size 128 --data_set HAR --kernel_size 3 --mu 100.0 --gumbel_tmp 1.0 --aug_mode cross_domain --gumbel_only False --beta 1 --alpha 1


# 2. finetune 
# python main_finetune.py --model_name  SATSC --exp_name aaai --batch_size 32 --data_set HAR  --subject 2 --file_name 1283 --file_epoch 100 --spt_lr 0.001 --device cuda --epoch 50 

