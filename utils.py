from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import os

class rag:
    def __init__(self, model_id="MLP-KTLim/llama-3-Korean-Bllossom-8B"):
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        ).to(self.device)

        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = Chroma(persist_directory="./.chroma_db", embedding_function=self.embedding_function)

    
    def store_data(self):
        path = "./originals/pdf/"
        for name in os.listdir(path=path):
            loader = PyPDFLoader(os.path.join(path, name))
            pages = loader.load_and_split()
            # split it into chunks
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs = text_splitter.split_documents(pages)
            Chroma.from_documents(docs, self.embedding_function, persist_directory="./.chroma_db")
        return True
    
    def answer_llm(self, user_prompt, system_prompt="You are an helpful assistant"):
        messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        )

        response = outputs[0][input_ids.shape[-1]:]
        
        return self.tokenizer.decode(response, skip_special_tokens=True)
    
    def answer_without_data(self, question):
        ans = self.answer_llm(user_prompt=question)
        return ans
    
    def answer_with_data(self, question):
        
        docs = self.db.similarity_search(question)
        context = docs[0].page_content

        system_prompt = "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use five sentences maximum and keep the answer concise."
        user_prompt = "\nQuestion: " + question + "\nContext: " + context +  "\nAnswer:"
        
        ans = self.answer_llm(user_prompt=user_prompt, system_prompt=system_prompt)
        
        return ans, docs[0]
    




