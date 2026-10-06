from llama_cpp import Llama
from vector_db import search
class LLMAssistant:
    def __init__(self, model_path="/home/roshan/models/gpt-oss/gpt-oss-20b-Q6_K.gguf"):
        self.ctx_size = 10000
        self.summary_ratio = 0.30
        self.System_prompt = (
        "You must respond naturally in spoken English. "
        "Never use lists, steps, bullets, numbering, or formatting. "
        "Speak in full conversational sentences only. "
        "If information is provided to you, you must use it to answer. "
    )



        self.Is_function_call_need_prompt = (
        "You are a decision engine for a call center system. "
        "Your job is to decide whether the user's message requires searching a database. "
        "Use search when the user asks about people, contact details, services, working hours, fees, procedures, or factual information. "
        "Use chat when the user is only greeting, confirming audio, or having normal conversation without requesting information. "
        "Rules: "
        "If search is needed, output in this format only: search:<query> "
        "The query must be a short, clear phrase suitable for database search. "
        "If search is not needed, output only: chat "
        "Do not explain. "
        "Do not use sentences. "
        "Do not add punctuation. "
        "Do not add extra text. "
        "Your output must be exactly one of these formats: chat OR search:<query>"
    )



        # Initialize model
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            main_gpu=0,
            n_ctx=self.ctx_size,
            verbose=False
        )
        
        # State-based chat history
        self.messages = [{"role": "system", "content": self.System_prompt}]
       


    def get_final_reply(self, user_input, system_instruction=None):

        # Inject temporary system message if present
        if system_instruction:
            self.messages.append({
                "role": "system",
                "content": system_instruction
            })
        self.messages.append({
            "role": "user",
            "content": user_input
        })


        response = self.llm.create_chat_completion(
            messages=self.messages,
            temperature=0.1,
            stream=False
        )
        print(f"messages sent to LLM: {self.messages}")
        full_content = response["choices"][0]["message"]["content"]

        # Remove the temporary system message so it doesn't persist
        if system_instruction:
            # remove the last system message we added
            for i in range(len(self.messages) - 1, -1, -1):
                if self.messages[i]["role"] == "system":
                    self.messages.pop(i)
                    break

        # Extract final answer
        if "<|channel|>final<|message|>" in full_content:
            final_answer = full_content.split("<|channel|>final<|message|>")[-1].strip()
        else:
            final_answer = full_content.strip()

        final_answer = final_answer.replace("<|end|>", "").strip()

        # Save assistant response (full, for context)
        self.messages.append({
            "role": "assistant",
            "content": full_content
        })
        print(f"LLM full response: {full_content}")

        return final_answer

    def check_for_function_calls(self, user_input, system_instruction=None):

        self.messages_for_function_calls = [
            {"role": "system", "content": self.Is_function_call_need_prompt}
        ]

        # Add user message to main conversation
        self.messages_for_function_calls.append({
            "role": "user",
            "content": user_input
        })


        response = self.llm.create_chat_completion(
            messages=self.messages_for_function_calls,
            temperature=0.1,
            stream=False
        )

        full_content = response["choices"][0]["message"]["content"]

        # Remove temporary system message from main history
        if system_instruction:
            for i in range(len(self.messages) - 1, -1, -1):
                if self.messages[i]["role"] == "system":
                    self.messages.pop(i)
                    break

        # Extract final answer
        if "<|channel|>final<|message|>" in full_content:
            final_answer = full_content.split("<|channel|>final<|message|>")[-1].strip()
        else:
            final_answer = full_content.strip()

        function_name = final_answer.replace("<|end|>", "").strip()
        print(f"Function call decision: {function_name}")

        return function_name

    
    def get_reply(self, user_input, system_instruction=None):
        function_name = self.check_for_function_calls(user_input, system_instruction)


        if function_name == "chat":
            return self.get_final_reply(user_input,system_instruction)
        elif function_name.startswith("search:"):
            query = function_name.replace("search:", "").strip()
            search_results = search(query)

            search_results = search(query)

            if search_results:
                # sort by score descending
                search_results = sorted(search_results, key=lambda x: x["score"], reverse=True)

                # take top 2 results
                top_results = search_results[:2]

                # extract only document text
                context_text = " ".join([r["document"].strip() for r in top_results])


                print(f"Filtered context: {context_text}")
                dynamic_system_prompt = (
                self.System_prompt +
                " The following information is available and must be used to answer: " +
                context_text
                )
                self.messages[0]["content"] = dynamic_system_prompt
                return self.get_final_reply(user_input)
                


                    

