from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama

from src.promt import refine_template 
from src.promt import prompt_template
def file_processing(file_path):
    loader = PyPDFLoader(file_path=file_path)
    data = loader.load()   

    question_gen = ""
    for page in data:
        question_gen += page.page_content

    splitter_ques_gen = TokenTextSplitter(
    model_name= "gpt-3.5-turbo",
    chunk_size= 10000,
    chunk_overlap = 200
)
    
    chunk_ques_gen = splitter_ques_gen.split_text(question_gen)


    from langchain.docstore.document import Document
    document_ques_gen = [Document(i) for i in chunk_ques_gen]

    splitter_ans_gen = TokenTextSplitter(
        model_name = 'gpt-3.5-turbo',
        chunk_size = 1000,
        chunk_overlap = 100
    )


    document_answer_gen = splitter_ans_gen.split_documents(
        document_ques_gen
    )

    return document_ques_gen, document_answer_gen   


def llm_pipeline(file_path):
    document_ques_gen, document_answer_gen = file_processing(file_path)
    llm = OllamaLLM(model="qwen2:1.5b",temperature=0.3)
    PROMTTEMPLATE = PromptTemplate(template=prompt_template ,input_variables=['text'])
    REFINE_PROMPT_QUESTIONS = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template,
    )
    ques_gen_chain = load_summarize_chain(llm=llm ,
                                          chain_type='refine', 
                                          verbose = True, 
                                          question_prompt=PROMTTEMPLATE  , 
                                          refine_prompt=REFINE_PROMPT_QUESTIONS)
    

    embeddings = OllamaEmbeddings(
    model="qwen2:1.5b",
)
    
    vector_store = FAISS.from_documents(documents=document_ques_gen, embedding=embeddings,)
    llm = ChatOllama(
    model="qwen2:1.5b",
    temperature=0.1,
)
    ques = ques_gen_chain.run(document_ques_gen)
    ques_list = ques.split("\n")
    filtered_ques_list = [element for element in ques_list if element.endswith('?') or element.endswith('.')]

    answer_generation_chain = RetrievalQA.from_chain_type(llm=llm, 
                                               chain_type="stuff", 
                                               retriever=vector_store.as_retriever())


    return answer_generation_chain, filtered_ques_list
















