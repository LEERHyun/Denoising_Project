# Denoising_Project

## 2025/03/07
### 모델 세부 구조 설정:
Level 1 Block: Transformer

Level 2,3 Block: NAFNet(CNN)

Middle: NAFBlock(CNN)


Num of Block

Encoder: [4,2,8]

Middle:12

Decoder: [4,2,2]

Refinement:12

## 2025/03/11
### 모델 세부 구조 구축

코드를 통해 각 모델(NAFNet, Restormer, Custom Model) 의 연산 복잡도(MACS), 크기(Parameter) 비교 가능

추후 진행 사항: 

1. NAFBLock의 Attrntion 모듈 개선(MDTA 활용 예정)

2. Matlab로 코드를 옮기기 위한 모듈화


