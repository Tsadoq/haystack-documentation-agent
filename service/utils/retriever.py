import os

from haystack.nodes import PreProcessor, WebRetriever


def return_retriever():
    preprocessor = PreProcessor(
        split_by="word",
        split_length=4096,
        split_respect_sentence_boundary=True,
        split_overlap=40,
    )

    return WebRetriever(
        api_key=os.environ['SERPERDEV_API_KEY'],
        allowed_domains=["docs.haystack.deepset.ai"],
        mode="preprocessed_documents",
        preprocessor=preprocessor,
        top_search_results=40,
        top_k=20,
    )
