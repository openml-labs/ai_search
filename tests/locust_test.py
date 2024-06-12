from locust import HttpUser, task, between
import json
import random

queries = ["Find datasets related to COVID-19", "Find datasets related to COVID-19 and India", "COVID-19 dataset", "COVID-19 dataset India", "Mexico historical covid", "Find me datasets related to mushrooms", "Fungi dataset", "Mushroom dataset", "shroom data", "types of mushroom", "earth fungus", "low features mushroom dataset", "plant datasets, low features", "plant, less number of features", "plant dataset, tiny"]

def sample_random_query(queries):
    return random.choice(queries)

class PerformanceTests(HttpUser):
    wait_time = between(5,10)

    @task(1)
    def test_dataset_flow(self):
        headers = {'Accept': 'application/json',
                   'Content-Type': 'application/json'}
        query = sample_random_query(queries)
        res = self.client.get(f"/dataset/{query}", 
                            #    data=json.dumps(sample.dict()),
                               headers=headers)
        # print("res",res.json())
