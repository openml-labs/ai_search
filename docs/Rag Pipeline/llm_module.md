# RAG LLM
- Setting up the retrival and using Lanchain APIs

## Modify LLM Chain
- At the moment the LLM chain is a retriver, if you want to add functionality, you will need to modify the `LLMChainInitializer` function.
- To change the way vectorstore is used, modify the `QASetup` function.
- To change the way Ollama works, caching works and add generation and stuff, modify the `LLMChainCreator` function. 

::: rag_llm