from llama_cpp import Llama

# Path to your GPT-OSS-20B model
MODEL_PATH = "/home/roshan/models/gpt-oss/gpt-oss-20b-Q4_0.gguf"

# Initialize the model with partial GPU offload
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=40,  # adjust to your VRAM; 46 may be too aggressive
    n_ctx=500,
    verbose=False
)

chat_history = []

print("GPT-OSS-20B Chatbot (type 'exit' to quit)")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        break

    # Build a clean, final-answer only prompt
    prompt = "You are a helpful AI assistant. Answer concisely and only with the final answer. Do not add explanations or repeated text.\n\n"
    
    for q, a in chat_history:
        prompt += f"User: {q}\nAssistant: {a}\n"
    
    prompt += f"User: {user_input}\nAssistant:"

    # Generate response
    output = llm(
        prompt,
        max_tokens=1028,          # safe for 16GB GPU
        temperature=0.0,         # deterministic, no randomness
        top_p=0.95,              # limits token variability
        repeat_penalty=1.2,      # penalize repetitions
        stop=["User:", "Assistant:"],
        echo=False
    )

    # Extract only the final answer
    response_text = output["choices"][0]["text"].strip().split("\n")[0]
    print(f"AI: {response_text}")

    chat_history.append((user_input, response_text))
