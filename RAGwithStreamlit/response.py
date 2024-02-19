from datasets import load_dataset
import pandas as pd
from operator import itemgetter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import DataFrameLoader
import os
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = "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"
def process():

    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )


    dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset", split='train')
    df = pd.DataFrame(dataset)
    loader = DataFrameLoader(df, page_content_column="Question")
    documents = loader.load()
    embeddings_model = OpenAIEmbeddings()
    db = FAISS.from_documents(documents, embeddings_model)
    retriever = db.as_retriever()
    template = """Answer the question strictly based only on the following context, if no context is give say 'I don't know':
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
    chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
    )
    return chain