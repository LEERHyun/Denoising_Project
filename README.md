# Denoising_Project

## 2025/03/07
### 모델 세부 구조 설정:
Level 1 Block: Transformer

Level 2,3 Block: NAFNet(CNN)

Middle: NAFBlock(CNN)


Num of Block

Encoder: [4,2,6]

Middle:12

Decoder: [4,2,2]

Refinement:12

## 2025/03/11
### 모델 세부 구조 구축

코드를 통해 각 모델(NAFNet, Restormer, Custom Model) 의 연산 복잡도(MACS), 크기(Parameter) 비교 가능

추후 진행 사항: 

   - 연산량이 높아져 효율적이지 않으므로, MDTA는 적용하지 않을 예정. 만약 한다면, 다른 Attention 모듈을 찾아볼 것

2. Matlab로 코드를 옮기기 위한 모듈화

## 2025/04/04
### Training 코드 추가


## 2025/04/14
### Model Editted version 추가

Encoder와 Decoder의 집합을 분리하여 각 Level별로 모듈을 분리

----------------------------------------------------------

코드 설명

