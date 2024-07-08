 
from langchain.chains.query_constructor.base import AttributeInfo
# from langchain_openai import ChatOpenAI
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)
from langchain_community.chat_models import ChatOllama

document_content_description = "metadata of machine learning datasets"
metadata_field_info = [
    AttributeInfo(
        name="description",
        description="A brief description about the dataset",
        type="string",
    ),
    AttributeInfo(
        name="id",
        description="dataset identifier",
        type="int",
    ),
    AttributeInfo(
        name="name",
        description="The name of the dataset",
        type="string",
    ),
    AttributeInfo(
        name="format",
        description="The format of available dataset files, such as ARFF, Parquet etc.",
        type="string",
    ),
    AttributeInfo(
        name="size",
        description="size of the dataset",
        type="int",
    )
]
prompt = get_query_constructor_prompt(
    document_content_description,
    metadata_field_info,
)

llm = ChatOllama(model="phi3", temperature=0)
output_parser = StructuredQueryOutputParser.from_components()
query_constructor = prompt | llm | output_parser

# print(prompt.format(query="mushroom datasets in ARFF format"))
print(query_constructor.invoke(
    {
        "query": "give mushroom datasets in ARFF format"
    }
))

print(query_constructor.invoke(
    {
        "query": "give me a mushroom dataset with 10k rows"
    }
))

