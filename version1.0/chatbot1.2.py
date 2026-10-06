from llama_cpp import Llama

MODEL_PATH = "/home/roshan/models/gpt-oss/gpt-oss-20b-Q4_0.gguf"
CTX_SIZE = 500
SUMMARY_RATIO = 0.30  # 30% of context

llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=46,
    n_ctx=CTX_SIZE,
    verbose=False
)

# Initial system prompt
BASE_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Provide accurate and direct answers."
)

messages = [
    {"role": "system", "content": BASE_SYSTEM_PROMPT}
]


def summarize_history(old_messages):
    """
    Summarize chat history into ~30% of context window.
    """
    summary_prompt = [
        {
            "role": "system",
            "content": (
                "Summarize the following conversation concisely. "
                "Preserve important facts, user preferences, and unresolved questions."
            )
        },
        {
            "role": "user",
            "content": "\n".join(
                f"{m['role'].upper()}: {m['content']}"
                for m in old_messages
                if m["role"] != "system"
            )
        }
    ]

    result = llm.create_chat_completion(
        messages=summary_prompt,
        max_tokens=int(CTX_SIZE * SUMMARY_RATIO),
        temperature=0.3
    )

    return result["choices"][0]["message"]["content"].strip()


def estimate_token_pressure():
    """
    Rough heuristic: llama_cpp throws error near ctx limit.
    We trigger summarization early.
    """
    return len(str(messages)) > 0.7 * CTX_SIZE


def chat_with_gpt_oss(user_input):
    global messages

    # Add user message
    messages.append({"role": "user", "content": user_input})

    # If context is getting large, summarize
    if estimate_token_pressure():
        summary = summarize_history(messages)

        messages = [
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {
                "role": "system",
                "content": f"Conversation summary so far:\n{summary}"
            },
            {"role": "user", "content": user_input}
        ]

    # Generate response
    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.7
    )

    full_content = response["choices"][0]["message"]["content"]
    
    # Logic to split reasoning from the final answer
    if "<|channel|>final<|message|>" in full_content:
        # Splits and takes the part after the 'final' tag
        final_answer = full_content.split("<|channel|>final<|message|>")[-1].strip()
    else:
        final_answer = full_content.strip()
        
    # Clean up trailing end tags if they appear
    final_answer = final_answer.replace("<|end|>", "").strip()
    
    messages.append({"role": "assistant", "content": full_content})
    return final_answer



# Interactive loop
while True:
    prompt = input("You: ")
    if prompt.lower() in ("exit", "quit"):
        break

    print("GPT-OSS:", chat_with_gpt_oss(prompt))
