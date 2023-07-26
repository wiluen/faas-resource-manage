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

handler = logging.FileHandler("/home/user/code/faas-resource/locust_log/locustfile_cancel.log")
handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logger = logging.getLogger("Debugging logger")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

class MyUser(HttpUser):
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
        
        response_order_refresh = self.client.post(
            url="http://33.33.33.99:31112/function/query-orders-for-refresh",
            headers=head,
            json=start_json
            )
       
        # 查询所有没付钱的列表  status:0没付钱 ， 4已取消  1已付钱  3可能是已付钱+已取消
        self.PAY_LIST=[]
        response_as_json=response_order_refresh.json()["data"]
        for order in response_as_json:
            if order["status"]==0 or order["status"]==1:
                self.PAY_LIST.append(order["id"])

        self.wait_to_cancel=random.choice(self.PAY_LIST)    #全局共享
        head = {"Accept": "application/json",
                "Content-Type": "application/json"}
        urls="http://33.33.33.99:32677/function/calculate-refund.openfaas-fn/orderId/"+self.wait_to_cancel
        start_time=time.time()
        response_refund=self.client.get(
            url=urls,
            headers=head
        )
        # end_time1=time.time()
        # to_log={'start':start_time1,'end':end_time1,'response_time':end_time1-start_time1,'response':self.try_to_read_response_as_json(response_refund)}
        # self.log_verbose(to_log)

        head = {"Accept": "application/json",
                "Content-Type": "application/json"}
        urls="http://33.33.33.99:32677/function/cancel-ticket.openfaas-fn/orderId/"+self.wait_to_cancel+"/loginId/4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"
        # start_time2=time.time()
        response_cancel=self.client.get(
            url=urls,
            headers=head
        )
        end_time=time.time()
        to_log={'response_time':end_time-start_time,'response':self.try_to_read_response_as_json(response_cancel)}
        self.log_verbose(to_log)
