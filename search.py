from locust import HttpUser, between, task
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
handler = logging.FileHandler("/home/user/code/faas-resource/locustfile_search.log")
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
    def search_ticket(self):
        TRIP=[{"from": "Shang Hai", "to": "Su Zhou"}]
                #  {"from": "Su Zhou", "to": "Shang Hai"},
                #  {"from": "Wu Xi", "to": "Shang Hai"},
                #  {"from": "Nan Jing", "to": "Shang Hai"},
                #  {"from": "Wu Xi", "to": "Su Zhou"}]
        TRAVEL_DATES = ["2023-07-30", "2023-08-01"]
        date=random.choice(TRAVEL_DATES)
        tripinfo=random.choice(TRIP)
        head = {"Accept": "application/json",
                "Content-Type": "application/json"}
        body_start = {
            "startingPlace": tripinfo['from'],
            "endPlace": tripinfo['to'],
            "departureTime": date
        }
        start_time = time.time()
        response = self.client.post(
            url="http://33.33.33.132:31112/function/get-left-trip-tickets",
            headers=head,
            json=body_start,
            )
        end_time=time.time()
        to_log = {'status_code': response.status_code,'state_time':start_time,'end_time':end_time,
                  'response_time': end_time - start_time}
        #           'response': self.try_to_read_response_as_json(response)}
        # to_log = {'status_code': response.status_code,
                #   'response_time': end_time - start_time}        
        self.log_verbose(to_log)
        
