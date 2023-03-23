# IMP_GCN with Contrastive Learning for Recommendation

## Environment Settings
- Tensorflow-gpu version:  1.3.0
- CUDA: 11.2.2
- Python: 3.7.16

## Example to run the codes.

# gowalla
Run IMP_GCN.py
```
python IMP_GCN.py --dataset gowalla  --cl_reg_s 0.07 --cl_reg_i 0.001 --regs [1e-4] --embed_size 64 --layer_size [64,64,64,64,64,64] --lr 0.001 --batch_size 2048 --epoch 2000 --groups 3 --Ks [20,10] --gpu_id 0
```