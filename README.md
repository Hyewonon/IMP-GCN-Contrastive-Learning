# IMP-GCN with Contrastive Learning for Recommendation

This is the Tensorflow implementation for KCC 2023 paper :
> 장혜원, 황혜수.(2023). 추천 시스템을 위한 관심사 인지 그래프 합성곱 신경망의 대조 학습 기법. 한국정보과학회 학술발표논문집, 제주. <br>
https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11488128-

## Introduction
IMP-GCN(Interest-aware message passing GCN) 모델은 기존의 GCN 기반 모델에서 발생하는 지나친 획일화(over-smoothing) 문제를 해결하기 위해, 사용자의 관심사에 따라 분류된 서브그래프 내에서 그래프 합성곱 연산을 수행한다. 하지만 IMP-GCN 모델에 몇 가지 한계점이 있다. 본 논문은 IMP-GCN의 한계점을 극복하고 성능을 향상시키기 위한 새로운 접근법을 제시한다. 제안 모델은 서브그래프 간 상품 수준과 서브그래프 수준의 대조적 학습을 활용하여, IMP-GCN의 상품 임베딩 산출과 서브그래프 분류 과정을 개선한다. 실험을 통해 제안 모델이 기존 모델보다 추천 성능을 향상시킬 수 있음을 확인했다.

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
