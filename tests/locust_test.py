"""
Load test to test the performance of the API using locust.
How to run:
- Start the FastAPI server using `uvicorn main:app`
- Load testing using Locust (locust -f tests/locust_test.py --host http://127.0.0.1:8000 ) using a different terminal
- Open the interface and configure the number of users and requests per second. Then, start the test.
- The test will run for the specified time and show the results.
"""
import random

from locust import HttpUser, between, task

queries: list[str] = [
    "Find datasets related to COVID-19",
    "Find datasets related to COVID-19 and India",
    "COVID-19 dataset",
    "COVID-19 dataset India",
    "Mexico historical covid",
    "Find me datasets related to mushrooms",
    "Fungi dataset",
    "Mushroom dataset",
    "shroom data",
    "types of mushroom",
    "earth fungus",
    "low features mushroom dataset",
    "plant datasets, low features",
    "plant, less number of features",
    "plant dataset, tiny",
]


def sample_random_query(queries : list[str]) -> str:
    """
    Description: Sample a random query from the list of queries
    
    Input: queries (List[str])
    
    Returns: query (str)
    """
    return random.choice(queries)


class PerformanceTests(HttpUser):
    wait_time = between(5, 10)

    @task(1)
    def test_dataset_flow(self) -> None:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        query = sample_random_query(queries)
        res = self.client.get(
            f"/dataset/{query}",
            #    data=json.dumps(sample.dict()),
            headers=headers,
        )
