from llama_cpp import Llama

llm = Llama(
    model_path="./llama-7b.Q4_K_M.gguf",
    n_gpu_layers=20,
)

prompt = """You are an AI assistant for a call center.
Answer questions clearly and factually.

Question: who you are. can i talk to a human?
Answer:"""

output = llm(
    prompt,
    max_tokens=128,
    temperature=0.5,
    top_p=0.5,
    repeat_penalty=1,
    stop=["Question:", "\n\n"],
    echo=False
)

print(output["choices"][0]["text"].strip())
