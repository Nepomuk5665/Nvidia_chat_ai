from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Load the trained model and tokenizer
model_dir = 'nepomuk'
model = GPTNeoForCausalLM.from_pretrained(model_dir)
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)


def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-processing: truncate at the first period
    if '.' in response:
        response = response.split('.')[0] + '.'

    # Post-processing: if the response is identical to the input, generate a new response
    if response.strip() == prompt.strip():
        return generate_response(prompt)

    return response


# Start the chatbot
print("Chatbot: Hello! How can I assist you?")

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    response = generate_response(user_input)
    print("Chatbot:", response)
