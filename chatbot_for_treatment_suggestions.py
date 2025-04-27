import os
from langchain_community.llms import CTransformers


os.environ["HF_HOME"] = r"C:\Users\A.C\.cache\huggingface"

def load_llm():
    model_path = r"C:\Users\A.C\.cache\huggingface\hub\models--TheBloke--Llama-2-7B-Chat-GGML\snapshots\76cd63c351ae389e1d4b91cab2cf470aab11864b\llama-2-7b-chat.ggmlv3.q2_K.bin"
    return CTransformers(model=model_path, model_type="llama", max_new_tokens=600, temperature=0.3, repetition_penalty=1.1)

def get_answer(query):
    """
    Uses AI to generate a structured medical response without relying on book data.
    """
    llm = load_llm()

    prompt = f"""You are a highly skilled medical AI.
Your job is to provide **detailed, medically accurate, and structured responses** to user questions.
Base your answers on scientific knowledge and best medical practices and behave like the you are
 presenter for the project named as computer aided diagnostic system using cnn for adhd and its subtypes

### **User's Medical Question:**
{query}

### **Final Professional Medical Answer:**
"""


    response = llm.invoke(prompt) 

    return response.strip()

if __name__ == '__main__':
     
    input_query = "Give treatment suggestions for  ADHD Subtypes diagnosed through MRI scan?"
    answer = get_answer(input_query)
    print(answer)