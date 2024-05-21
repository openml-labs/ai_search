
from modules.llm import *
from modules.utils import *

metadata_df = create_metadata_dataframe(*get_all_dataset_metadata_from_openml())
metadata_df = clean_metadata_dataframe(metadata_df)

print(metadata_df.loc[20]['description'][:300])

vectordb = load_document_and_create_vector_store(metadata_df)
# vectordb = load_document_and_create_vector_store(metadata_df, recreate_chroma=True)

rqa_prompt_template = "This database is a list of dataset metadata. Use the following pieces of context to find the relevant document. Answer only from the context given using the {question} given. If you do not know the answer, say you do not know. {context}"

retriever, llm = create_retriever_and_llm(vectordb)
qa = create_llm_chain_and_query(vectordb=vectordb,retriever=retriever,llm=llm, prompt_template = rqa_prompt_template)

query = "Which datasets would be useful for stock market information?"

results = create_result_dataframe(query, qa)
print(results)