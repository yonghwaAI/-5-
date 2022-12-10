# AIFT team5 

* 현재 진행 상황 … 미정
* 이슈상황 … 전부
* 목표
 1. 평균회귀전략
 2. PER PBR 등 주가지표 이용
 3. 머신러닝
 
 ## main()
 - rsi_strategy = RSIStrategy() : sub thread 생성 
 - rsi_strategy.start()         : sub thread의 run 메소드 호출

## Modeling - flaml()
 - task - 목표로하는 작업. 'classification', 'regression', 'ts_forecast', 'rank', 'seq-classification', 'seq-regression', 'summarization'
estimator_list 
 - 사용할 모델의 종류. 'auto'를 선택할 수 있음. 이외 직접 선택할 수 있는 목록: 'lgbm', 'xgboost', 'xgb_limitdepth', 'catboost', 'rf', 'extra_tree'
time_budget 
 - 시간에 대한 제약조건. '-1' 입력 시, 무제한

 - xgboost:(accuracy = 0.8458850931677019)
   - learning curve: 
      <img src="https://user-images.githubusercontent.com/90076289/206838013-c392d429-67a9-4c8f-a0d3-378f60126232.png" width="400" height="200"/>
   - 혼동행렬:
      <img src="https://user-images.githubusercontent.com/90076289/206838046-141eb9ae-06b2-4450-a4ae-28218c97ad99.png" width="400" height="200"/>
     
 - random forest(accuracy = 0.7472826086956522)
   - learning curve: 
       <img src="https://user-images.githubusercontent.com/90076289/206837904-d4079625-e095-4947-b630-da95b9d7354d.png" width="400" height="200"/>
   - 혼동행렬:
       <img src="https://user-images.githubusercontent.com/90076289/206837908-2c28151e-102f-4465-bbac-190e1dc6cb2a.png" width="400" height="200"/>
       

