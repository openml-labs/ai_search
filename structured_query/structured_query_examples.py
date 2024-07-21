examples = [
    (
        "Give me mushroom datasets with less than 10k rows.",
        {
            "query": "mushroom dataset",
            "filter": 'lt("NumberOfInstances", 10000)',
        },
    ),
    (
        "Give me datasets than can be used in healthcare with 2 or more classes",
        {
            "query": "heathcare dataset ",
            "filter": 'gte("NumberOfClasses", 2.0)',
        },
    ),
    (
        "Give me datasets than can be used in healthcare, or climate applications with 2 or more classes and in arff format.",
        {
            "query": "heathcare dataset, climate datasets",
            "filter": 'and(gte("NumberOfClasses", 2.0), eq("format", "arff"))',
        },
    ),
    (
        "Give me medical datasets.",
        {
            "query": "medical datasets",
            "filter": "NO_FILTER",
        },
    ),
    (
        "Give me medical datasets with large number of features.",
        {
            "query": "medical datasets",
            "filter": 'gte("NumberOfFeatures, 100)',
        },
    ),
]
