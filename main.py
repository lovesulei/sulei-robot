from brain.llm import LLM

brain = LLM()

while True:

    message = input("You: ")

    if message.lower() == "quit":
        break

    print()
    print("Robot:")
    print(brain.chat(message))
    print()