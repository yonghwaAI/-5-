from api.Kiwoom import *
from util.db_helper import *
from util.time_helper import *
from model.ml_model import *
from predict.predicti import InputBuilder_BaselineModel, BaselineModel
import math
import traceback
import sys
import numpy as np
import pandas as pd


class Strategy(QThread):
    def __init__(self):
        QThread.__init__(self)
        self.strategy_name = "Strategy"
        self.kiwoom = Kiwoom()
        self.tech = Tech_model()
        #self.predict = BaselineModel()
        self.no_record = True

        # 계좌 예수금
        self.deposit = 0

        # 초기화 함수 성공 여부 확인 변수
        self.is_init_success = False

        self.init_strategy()

    def init_strategy(self):
        """전략 초기화 기능을 수행하는 함수"""
        try:
            # 유니버스 조회, 없으면 생성
            self.check_and_get_universe()

            # 가격 정보를 조회, 필요하면 생성
            self.check_and_get_price_data()

            # Kiwoom > 주문정보 확인
            self.kiwoom.get_order()

            # Kiwoom > 잔고 확인
            self.kiwoom.get_balance()

            # Kiwoom > 예수금 확인
            self.deposit = self.kiwoom.get_deposit()

            # predict > input_value
            # self.input = InputBuilder_BaselineModel(self.universe)

            # 유니버스 실시간 체결정보 등록
            self.set_universe_real_time()

            self.is_init_success = True

        except Exception as e:
            print(traceback.format_exc())
    
     # 실험
    def check_and_get_universe(self):
        now = datetime.now().strftime("%Y%m%d%H%M")

        self.universe = {'069500':'kodex_200', '114800':'kodex_inverse'}

        universe_list= [['069500', 'kodex_200'], ['114800','kodex_inverse']]

        self.universe_df = pd.DataFrame({
                'code': self.universe.keys(),
                'code_name': self.universe.values(),
                'created_at': [now] * len(self.universe.keys())
        })

        # universe라는 테이블명으로 Dataframe을 DB에 저장함
        insert_df_to_db(self.strategy_name, 'universe', self.universe_df)

        sql = "select * from universe"
        cur = execute_sql(self.strategy_name, sql)
        universe_list = cur.fetchall()
        for item in universe_list:
            idx, code, code_name, created_at = item
            self.universe[code] = {
                'code_name': code_name
            }
        print(self.universe)

    def check_and_get_price_data(self):
        """분봉 데이터가 존재하는지 확인하고 없다면 생성하는 함수"""
        for idx, code in enumerate(self.universe.keys()):
            print("({}/{}) {}".format(idx + 1, len(self.universe), code))

            # (1)케이스: 분봉 데이터가 아예 없는지 확인(장 종료 이후)
            if check_transaction_closed() and not check_table_exist(self.strategy_name, code):
                print("(1)분봉 데이터가 없음")
                # API를 이용해 조회한 가격 데이터 price_df에 저장
                price_df = self.kiwoom.get_price_data(code)
                # 코드를 테이블 이름으로 해서 데이터베이스에 저장
                insert_df_to_db(self.strategy_name, code, price_df)
                # 가격 데이터를 self.universe에서 접근할 수 있도록 저장
                self.universe[code]['price_df'] = price_df
            else:
                # (2-1), (2-2), (2-3) 케이스: 일봉 데이터가 있는 경우
                # (2-1)케이스: 장이 종료된 경우 API를 이용해 얻어온 데이터를 저장
                if check_transaction_closed():
                    print("(2-1)장이 종료된 경우")
                    # 저장된 데이터의 가장 최근 일자를 조회
                    sql = "select max(`{}`) from `{}`".format('index', code)

                    cur = execute_sql(self.strategy_name, sql)

                    # 분봉 데이터를 저장한 가장 최근 일자를 조회
                    last_date = cur.fetchone()

                    # 오늘 날짜를 20210101 형태로 지정
                    now = datetime.now().strftime("%Y%m%d%H%M")

                    # -------------------------------------------
                    # 데이터베이스에 저장된 데이터 조회
                    sql = "select * from `{}`".format(code)
                    cur = execute_sql(self.strategy_name, sql)
                    cols = [column[0] for column in cur.description]
                    # 데이터베이스에서 조회한 데이터를 DataFrame으로 변환해서 저장
                    price_df = pd.DataFrame.from_records(data=cur.fetchall(), columns=cols)
                    price_df = price_df.set_index('index')
                    # 가격 데이터를 self.universe에서 접근할 수 있도록 저장
                    self.universe[code]['price_df'] = price_df
                    # -------------------------------------------


                    # 최근 저장 일자가 오늘이 아닌지 확인
                    if last_date[0] != now:
                        print(last_date[0])
                        print("최근 저장 일자가 오늘이 아님")
                        price_df = self.kiwoom.get_price_data(code)
                        # 코드를 테이블 이름으로 해서 데이터베이스에 저장
                        insert_df_to_db(self.strategy_name, code, price_df)
                        # 가격 데이터를 self.universe에서 접근할 수 있도록 저장
                        self.universe[code]['price_df'] = price_df

                # (2-2)케이스: 장 시작 전이거나 장 중인 경우 데이터베이스에 저장된 데이터 조회
                else:
                    print("(2-2)장 시작 전이거나 장 중인 경우")
                    sql = "select * from `{}`".format(code)
                    cur = execute_sql(self.strategy_name, sql)
                    cols = [column[0] for column in cur.description]

                    # 데이터베이스에서 조회한 데이터를 DataFrame으로 변환해서 저장
                    price_df = pd.DataFrame.from_records(data=cur.fetchall(), columns=cols)
                    price_df = price_df.set_index('index')
                    # 가격 데이터를 self.universe에서 접근할 수 있도록 저장
                    self.universe[code]['price_df'] = price_df
        
###########
    
    def run(self):
        """실질적 수행 역할을 하는 함수"""
        print('\n', self.universe['069500']['price_df'])
        print('------------------------------------------------------------------------------------------------- \n')
        # df = self.tech.get_daily_dic(self.universe)
        
        input_builder = InputBuilder_BaselineModel(self.universe)
        y_pred = BaselineModel().predict(input_builder.X_test)
        print('예측결과 : ', y_pred)

    #     # 접주 주문 및 매수/매도 대상 확인
    #     while self.is_init_success:
    #         try:
    #             # (0)장중인지 확인
    #             if not check_transaction_open():
    #                 print("장시간이 아니므로 종료합니다.")
    #                 sys.exit()

    #             for idx, code in enumerate(self.universe.keys()):
    #                 print('[{}/{}_{}]'.format(idx + 1, len(self.universe), self.universe[code]['code_name']))
    #                 time.sleep(0.5)

    #                 # (1)접수한 주문이 있는지 확인
    #                 if code in self.kiwoom.order.keys():
    #                     # (2)주문이 있음
    #                     print('접수 주문', self.kiwoom.order[code])

    #                     # (2.1) '미체결수량' 확인하여 미체결 종목인지 확인
    #                     if self.kiwoom.order[code]['미체결수량'] > 0:
    #                         pass

    #                 # (3)보유 종목인지 확인
    #                 elif code in self.kiwoom.balance.keys():
    #                     print('보유 종목', self.kiwoom.balance[code])
    #                     # (6)매도 대상 확인
    #                     if self.check_sell_signal(code):
    #                         # (7)매도 대상이면 매도 주문 접수
    #                         self.order_sell(code)

    #                 else:
    #                     # (4)접수 주문 및 보유 종목이 아니라면 매수대상인지 확인 후 주문접수
    #                     self.check_buy_signal_and_order(code)

    #         except Exception as e:
    #             print(traceback.format_exc())
            
    #         # 매매할 때마다(체결이 됐을 때) 얼마에 사고 얼마에 팔았는지, 예수금, 수익률 출력, 기록
    #         # 장 끝날 때도 출력
            
    #         # 초기값 200을 하든 inverse를 하든 매수만
    #         # 데이터 들어올 때마다 모델에 넣고 예측
    #         # ㄴ X=(kodex_200 매수 & kodex_inverse 매도), Y=(kodex_200 매도 & kodex_inverse 매수), NOP=Nothing
            
    #         # 30분 / 1시간 단위로 계속 자동투자를 진행할 것인지 물어볼까? / 중간에 빠져나오거나(그만)
    #         # 틱범위 1분 어떨까
    #         # target 95% → 90%


    def set_universe_real_time(self):
        """유니버스 실시간 체결정보 수신 등록하는 함수"""
        # 임의의 fid를 하나 전달하기 위한 코드(아무 값의 fid라도 하나 이상 전달해야 정보를 얻어올 수 있음)
        fids = get_fid("체결시간")

        # universe 딕셔너리의 key값들은 종목코드들을 의미
        codes = self.universe.keys()

        # 종목코드들을 ';'을 기준으로 묶어주는 작업
        codes = ";".join(map(str, codes))

        # 화면번호 9999에 종목코드들의 실시간 체결정보 수신을 요청
        self.kiwoom.set_real_reg("9999", codes, fids, "0")
        

    def return_real_data(self, code):
        universe_item = self.universe[code]

        # (1)현재 체결정보가 존재하지 않는지 확인
        if code not in self.kiwoom.universe_realtime_transaction_info.keys():
            # 존재하지 않다면 더이상 진행하지 않고 함수 종료
            print("매수대상 확인 과정에서 아직 체결정보가 없습니다.")
            return

        # (2)실시간 체결 정보가 존재하면 현 시점의 시가 / 고가 / 저가 / 현재가 / 누적 거래량이 저장되어 있음
        open = self.kiwoom.universe_realtime_transaction_info[code]['시가']
        high = self.kiwoom.universe_realtime_transaction_info[code]['고가']
        low = self.kiwoom.universe_realtime_transaction_info[code]['저가']
        close = self.kiwoom.universe_realtime_transaction_info[code]['현재가']
        volume = self.kiwoom.universe_realtime_transaction_info[code]['누적거래량']

        # 오늘 가격 데이터를 과거 가격 데이터(DataFrame)의 행으로 추가하기 위해 리스트로 만듦
        today_price_data = [open, high, low, close, volume]

        df = universe_item['price_df'].copy()

        # 과거 가격 데이터에 금일 날짜로 데이터 추가
        df.loc[datetime.now().strftime('%Y%m%d%H%M')] = today_price_data

        return df
    
    def order_sell(self, code):
        """매도 주문 접수 함수"""
        # 보유 수량 확인(전량 매도 방식으로 보유한 수량을 모두 매도함)
        quantity = self.kiwoom.balance[code]['보유수량']
        # 최우선 매도 호가 확인
        ask = self.kiwoom.universe_realtime_transaction_info[code]['(최우선)매도호가']
        order_result = self.kiwoom.send_order('send_sell_order', '1001', 2, code, quantity, ask, '00')
        print('\n 매도주문: ',order_result)
    
    def order_buy(self, code):
        """매수 주문 접수 함수"""
        print('\n 매수 전 예수금: ', self.deposit)
        # (1)주문에 사용할 금액 계산
        budget = self.deposit
        # (2)최우선 매수호가 확인
        bid = self.kiwoom.universe_realtime_transaction_info[code]['(최우선)매수호가']
        # (3)주문 수량 계산(소수점은 제거하기 위해 버림)
        quantity = math.floor(budget / bid)
        # (4)현재 예수금에서 수수료를 곱한 실제 투입금액(주문 수량 * 주문 가격)을 제외해서 계산
        amount = quantity * bid
        self.deposit = math.floor(self.deposit - amount * 1.00015)
        # (5)예수금이 0보다 작아질 정도로 주문할 수는 없으므로 체크
        if self.deposit < 0:
            return
        # (6)계산을 바탕으로 지정가 매수 주문 접수
        order_result = self.kiwoom.send_order('send_buy_order', '1001', 1, code, quantity, bid, '00')
        print('\n 매수주문: ',order_result)
        # _on_chejan_slot가 늦게 동작할 수도 있기 때문에 미리 약간의 정보를 넣어둠
        self.kiwoom.order[code] = {'주문구분': '매수', '미체결수량': quantity}

    def check_buy_signal_and_order(self):
        """매수 대상인지 확인하고 주문을 접수하는 함수
        새 data를 universe로 받음, self.input에 넣고 X_test 값을 받음, model에 적용, 
        nop, x, y에 따라 매수 / 매도
        """
        # 매수 가능 시간 확인
        if not check_adjacent_transaction_closed():
            return False

        # 실시간 유니버스 만들기
        real_069500 = self.return_real_data('069500')
        real_114800 = self.return_real_data('114800')

        real_universe = self.universe.copy()
        real_universe['069500']['price_df'] = real_069500
        real_universe['114800']['price_df'] = real_114800

        ## predict > input_value
        input_builder = InputBuilder_BaselineModel(self.universe)
        predict_value = BaselineModel().predict(input_builder.X_test)
        # predict_proba = BaselineModel().predict_proba(input_builder.X_test)

        # 매수 신호 확인(조건에 부합하면 주문 접수) ★★★★★★여기에 모델결과 적용★★★★★★
        if predict_value == 0:
            print('predict_value:',predict_value)
            pass
        elif predict_value == 1: # 1=(kodex_200 매수 & kodex_inverse 매도)
            print('predict_value:',predict_value)
            if self.no_record:
                ## 전부 매수 
                self.order_buy('069500')
                self.no_record = False
                pass
            else:
                # 전부 매도
                self.order_sell('114800')
                # 전부 매수
                self.order_buy('069500')
                pass
        elif predict_value == 2: # 2=(kodex_200 매도 & kodex_inverse 매수)
            print('predict_value:',predict_value)
            if self.no_record:
                # 전부 매수
                self.order_buy('114800')
                self.no_record = False
                pass
            else:
                # 전부 매도
                self.order_sell('069500')
                # 전부 매수
                self.order_buy('114800')
                pass