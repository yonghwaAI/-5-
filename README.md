# 인금투 

* 현재 진행 상황 … 미정
* 이슈상황 … 전부
* 목표
 1. 평균회귀전략
 2. PER PBR 등 주가지표 이용
 3. 머신러닝
 
 ## main()
 - rsi_strategy = RSIStrategy() : sub thread 생성 
 - rsi_strategy.start()         : sub thread의 run 메소드 호출

## flaml()
 - task - 목표로하는 작업. 'classification', 'regression', 'ts_forecast', 'rank', 'seq-classification', 'seq-regression', 'summarization'
estimator_list 
 - 사용할 모델의 종류. 'auto'를 선택할 수 있음. 이외 직접 선택할 수 있는 목록: 'lgbm', 'xgboost', 'xgb_limitdepth', 'catboost', 'rf', 'extra_tree'
time_budget 
 - 시간에 대한 제약조건. '-1' 입력 시, 무제한
