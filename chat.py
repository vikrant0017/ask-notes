import sys
from rag import initialize, ask

def chat_with_model():
    def get_dirname():
        length = len(sys.argv)
        dirname = 'notes' # default dir to use
        if length > 1:
            dirname = sys.argv[1]

        return dirname
            
    dirname = get_dirname()
    initialize(dirname)
    print("Hello, I am here to answer your queries based on your notes.")
    print("Type 'exit' to quit.")
    print()
    
    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            print("Exiting chat...")
            break
            
        # Simulate a response from an open-source model
        response = ask(user_query)
        print(f'Assistant: {response}')

if __name__ == "__main__":
    chat_with_model()
