# %%
import requests


# %%
def get_elastic_search_results(query):
    query = query.replace(" ", "%20")
    url = "https://es.openml.org/_search?q=" + query
    response = requests.get(url)
    response_json = response.json()
    return response_json["hits"]["hits"]


# %%
res = get_elastic_search_results("iris")
# %%
ids = [val["_id"] for val in res]
