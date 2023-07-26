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

TRIP_DATA = [{"from": "Shang Hai", "to": "Su Zhou", "trip_id": "D1345", "seat_type": "2", "seat_price": "50.0"},
             {"from": "Shang Hai", "to": "Su Zhou", "trip_id": "D1345", "seat_type": "3", "seat_price": "22.5"},
             {"from": "Su Zhou", "to": "Shang Hai", "trip_id": "G1237", "seat_type": "2", "seat_price": "50.0"},
             {"from": "Su Zhou", "to": "Shang Hai", "trip_id": "G1237", "seat_type": "3", "seat_price": "30.0"},
             {"from": "Nan Jing", "to": "Shang Hai", "trip_id": "G1234", "seat_type": "2", "seat_price": "250.0"},
             {"from": "Nan Jing", "to": "Shang Hai", "trip_id": "G1234", "seat_type": "3", "seat_price": "95.0"},
             {"from": "Nan Jing", "to": "Shang Hai", "trip_id": "G1235", "seat_type": "2", "seat_price": "250.0"},
             {"from": "Nan Jing", "to": "Shang Hai", "trip_id": "G1235", "seat_type": "3", "seat_price": "125.0"},
             {"from": "Nan Jing", "to": "Shang Hai", "trip_id": "G1236", "seat_type": "2", "seat_price": "250.0"},
             {"from": "Nan Jing", "to": "Shang Hai", "trip_id": "G1236", "seat_type": "3", "seat_price": "175.0"},
             {"from": "Wu Xi", "to": "Shang Hai", "trip_id": "G1234", "seat_type": "2", "seat_price": "100.0"},
             {"from": "Wu Xi", "to": "Shang Hai", "trip_id": "G1234", "seat_type": "3", "seat_price": "38.0"}]
CONTACT_ID = ["56ae6357-2751-4366-a2f6-ff6cd032b89c",
            "be82cba3-944a-433e-94d5-dae4d242bc13",
            "7ab77385-67ef-4bb6-b310-74f4063640d9",
            "c65a89cc-8e52-41a3-b607-e185e4fbeda4",
            "a85e8ab4-e20d-4dbd-8b93-2a023e0aa760",
            "d8fc1667-8614-4932-94df-e2ad72b2b6b3",
            "c0cdc685-6f75-4b4d-ba71-aa5829335a12"]
TRAVEL_DATES = [ "2023-07-2", "2023-07-8"]
account_id="4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"
contact=random.choice(CONTACT_ID)
trip_detail = random.choice(TRIP_DATA)
date=random.choice(TRAVEL_DATES)
dir_path = os.path.dirname(os.path.realpath(__file__))
handler = logging.FileHandler(os.path.join(dir_path, "locustfile_preserve.log"))
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
    def perserve_ticket(self):
        account_id="4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"
        contact=random.choice(CONTACT_ID)
        trip_detail = random.choice(TRIP_DATA)
        date=random.choice(TRAVEL_DATES)
        head = {"Accept": "application/json",
                "Content-Type": "application/json"}
        body_start = {
            "accountId": account_id,
            "contactsId": contact,
            "date": date,
            "foodType": 0,
            "from": trip_detail["from"],
            "seatType": trip_detail["seat_type"],
            "to": trip_detail["to"],
            "tripId": trip_detail["trip_id"]
        }
        # bodystr=json.dumps(body_start)
        # start_time = time.time()
        response = self.client.post(
            url="http://33.33.33.99:31112/function/preserve-ticket",
            headers=head,
            json=body_start,
            )
        # to_log={'json':body_start,'state_time':start_time,'response_time':time.time()-start_time,'response':self.try_to_read_response_as_json(response)}
        # self.log_verbose(to_log)
