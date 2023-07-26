from locust import HttpUser, between, task,SequentialTaskSet
import random
import sys
import time
import json
import numpy as np
from datetime import datetime
from random import randint
import logging
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
handler = logging.FileHandler(os.path.join(dir_path, "locustfile_pay.log"))
handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logger = logging.getLogger("Debugging logger")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

USERID="4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"
class MyTaskSet(SequentialTaskSet):
    wait_time = between(1, 3)

    def log_verbose(self, to_log):
        logger.debug(json.dumps(to_log))

    def try_to_read_response_as_json(self, response):
        try:
            return response.json()
        except:
            try:
                return response.content.decode('utf-8')
            except:
                return response.content
            
    @task
    def get_order(self):
        head = {"Accept": "application/json",
                "Content-Type": "application/json"}
        start_json={
            "loginId": "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f",
            "enableStateQuery": "false",
            "enableTravelDateQuery": "false",
            "enableBoughtDateQuery": "false",
            "travelDateStart": "null",
            "travelDateEnd": "null",
            "boughtDateStart": "null",
            "boughtDateEnd": "null"
            }
        # 随着数据传输的越来越多，这个请求延迟变慢
        response_order_refresh = self.client.post(
            url="http://33.33.33.99:31112/function/query-orders-for-refresh",
            headers=head,
            json=start_json
            )
        # 查询所有没付钱的列表
        self.PAY_LIST=[]
        response_as_json=response_order_refresh.json()["data"]
        for order in response_as_json:
            if order["status"]==0:
                self.PAY_LIST.append({"orderid":order["id"],"tripid":order["trainNumber"]})

                # status:0没付钱 ， 4已取消  1已付钱  3可能是已付钱+已取消
# {"orderId":"6b577da4-ca29-45c4-b38a-2f3ce0bccb51","tripId":"G1237","userId":"4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"}
    @task
    def pay(self):
        wait_to_pay=random.choice(self.PAY_LIST)
        head = {"Accept": "application/json",
                "Content-Type": "application/json"}
        star_json={
            "orderId":wait_to_pay["orderid"],
            "tripId":wait_to_pay["tripid"],
            "userId":USERID
        }
        start_time=time.time()
        response_pay=self.client.post(
            url="http://33.33.33.99:31112/function/pay-for-the-order",
            headers=head,
            json=star_json
        )
        to_log={'json':star_json,'response_time':time.time()-start_time,'response':self.try_to_read_response_as_json(response_pay)}
        self.log_verbose(to_log)

        
        
class MyUser(HttpUser):
    wait_time = between(1, 3)
    tasks = [MyTaskSet]

# 设置了随机支付订单，目前看起来没什么问题，相比较每次支付PAY_LIST[-1]，只能一个用户成功支付
