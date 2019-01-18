# Chapter 8. Dimensionality Reduction(차원 축소)

## Curse of Dimensionality & Dimensionality Reduction

### 차원의 저주
- feature이 많을 수록 학습하는데 많은 시간 소요
- 좋은 solution 찾기 어렵다.
- 고차원일수록 instance간 거리가 너무 멀어 예측 신뢰도가 떨어진다.(less reliable)
- 차원이 클수록 데이터셋이 sparse해지고, 모델이 과적합(overfitting)될 가능성이 크다.
   
### 차원 축소의 목적
- feature들을 대폭 줄이는 것이 가능하다. 
- data visualization에 유용하다. 고차원 데이터를 2(간혹 3)차원으로 낮춰 시각적으로 패턴을 분석하여 통찰 얻을 수 있다.

## Main Approaches for Dimensionality Reduction
### Projection
실세계에서 training instance는 모든 차원에서 균등하게 분포되어있진 않다.

