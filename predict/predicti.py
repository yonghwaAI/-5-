import pickle

# class Predict(prediction_pb2_grpc.PredictorServicer):
#     def __init__(self, model):
#         self.model=model

#     def Predict(self, request, context):
#         # dummy implementation for just testing
#         # value 불러오기
#         input_builder = InputBuilder_BaselineModel(request)
#         # 값 예측, 저장
#         y_pred = self.model.predict(input_builder.X_test)
#         # 확률 예측, 저장
#         y_proba = self.model.predict_proba(input_builder.X_test)

#         return prediction_pb2.PredictResponse(actions={'NOP':y_proba[0][0], 'X':y_proba[0][1], 'Y':y_proba[0][2]})


# def serve(model):
#     server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
#     prediction_pb2_grpc.add_PredictorServicer_to_server(PredictionServer(model), server)
#     server.add_insecure_port('[::]:50051')
#     server.start()
#     server.wait_for_termination()

# if __name__ == "__main__":
#     # cm = ConfigManager('config/.config.xml')
#     # with open( "./jupyter/.grpc_reqest_sample.pkl", "rb" ) as file:
#     #     serialized_buf = pickle.load(file)
#     # req = prediction_pb2.PredictRequest.FromString(serialized_buf)
#     # input_builder = InputBuilder_BaselineModel(req)
#     # x = input_builder.X_test
#     # print(x)

#     model = BaselineModel("./jupyter/.automl_3m.pkl")

#     logging.basicConfig()
#     logging.getLogger().setLevel(logging.INFO)
#     serve(model)


# def return_predict_value(value:int):
#     '''load pickled automl object'''
#     with open('automl_flaml_xgboost.pkl', 'rb') as f:
#         automl = pickle.load(f)

#     '''compute predictions of testing dataset''' 
#     y_pred = automl.predict(X_test)
#     # print('Predicted labels', y_pred)
#     # print('True labels', y_test)
#     y_pred_proba = automl.predict_proba(X_test)


with open('predict/automl_baseline2_balanced_10m.pkl', 'rb') as f:
    model = pickle.load(f)
