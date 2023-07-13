# Implement a RAG solution for Video/Audio data

The existing LLM solution such as RAG or ChatBot only support text data sources. However, video/audio data is also one of the most important knowledge base for organizations holding massive media data. In addition, compared to text data such as documents or books, it's harder to look up information from video/audio data. People may have to go through all video/audio file to localize the information they need. 

In this project, we provide video and audio processing solution for adopting generative AI on video and audio data. There are two main scenarios, 1) Enterprise can enrich their knowledge base with the existing video/audio data, which can make RAG more possible to get relevant information from knowledge base. 2) Individual users can efficiently get the informaiton they are interested in and reach to the most relevant localtion in the video/audio file, which can save much time to look up the information.

We put our data into the following RAG solutions:
 - [Build a powerful question answering bot with Amazon SageMaker, Amazon OpenSearch Service, Streamlit, and LangChain](https://aws.amazon.com/blogs/machine-learning/build-a-powerful-question-answering-bot-with-amazon-sagemaker-amazon-opensearch-service-streamlit-and-langchain/)
 - [Question answering using Retrieval Augmented Generation with foundation models in Amazon SageMaker JumpStart](https://aws.amazon.com/blogs/machine-learning/question-answering-using-retrieval-augmented-generation-with-foundation-models-in-amazon-sagemaker-jumpstart/)
 
 
## Data preparation
Run [data_preparation.ipynb](data_preparation.ipynb)


## Inject data into RAG solution
- [Inject data into OpenSeach for RAG](data_ingestion_to_vectordb.ipynb)
- [Inject data into LangChain VectorDB for RAG](video_question_answering_langchai.ipynb)

## Application
We demonstrate [RAG with OpenSearch](./app_rag) on video/audio data and [chatbot](./app_chatbot) on video/audio data with steamlit.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.