# Testing

## Load Testing
- Load testing can be done using Locust, a load testing tool that allows you to simulate users querying the API and measure the performance of the API under load from numerous users.
- It is possible to configure the number of users, the hatch rate, and the time to run the test for.

## Running the load test
- Start the FastAPI server using `uvicorn main:app`
- Load testing using Locust (`locust -f tests/locust_test.py --host http://127.0.0.1:8000` ) using a different terminal