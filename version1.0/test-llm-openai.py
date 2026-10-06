from llama_cpp import Llama

llm = Llama(
    #model_path="/home/roshan/models/gpt-oss/gpt-oss-20b-Q4_0.gguf",
    model_path="/home/roshan/models/gpt-oss/gpt-oss-20b-Q5_K_M.gguf",
    n_gpu_layers=46,
    n_ctx=2048,
    verbose=True
)

prompt = """### Instruction:
Answer accurately.
Do not add explanations. only give the answer.

Question: can you help me to write a code?

### Response:
"""


output = llm(
    prompt,
    max_tokens=2000,          # MUST be > 0
    temperature=0.2,
    top_p=0.9,
    repeat_penalty=1.1,
    stop=["### Instruction:"],
    echo=False
)




raw_text = output["choices"][0]["text"]

marker = "<|end|><|start|>assistant<|channel|>final<|message|>"
if marker in raw_text:
    final_answer = raw_text.split(marker)[-1].strip()
else:
    final_answer = raw_text.strip()

print(raw_text)
