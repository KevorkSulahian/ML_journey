import openai
from typing import List, Dict
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv() 

class SupportAssistant:
    def __init__(self):
        # openai.api_key = api_key
        self.conversation_history = []
        self.current_ticket = {}
        self.client = OpenAI() 
        
    def add_to_history(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})
    
    def get_completion(self, prompt: str) -> Dict:
        self.add_to_history("user", prompt)
        
        messages = [
            {"role": "system", 
            "content": """You are a helpful support assistant. 
            Classify user requests into: Bug, Feature Request, or Other. 
            Collect required information based on the type:
            - Bug: username, description, version
            - Feature Request: username, description, priority
            - Other: username, description.
            
            Return the classification and details in JSON format.
            If there are information missing, ask for it.
            and don't end the conversation is there are information missing.
            If all of the required information is provided return with {"end_conversation": true}."""},
            *self.conversation_history
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.2,
            max_tokens=150
        )
        
        raw_response = response.choices[0].message.content
        self.add_to_history("assistant", raw_response)

        # Try to parse JSON safely
        try:
            parsed_response = json.loads(raw_response)
        except json.JSONDecodeError:
            parsed_response = {"message": raw_response, "end_conversation": False}

        return parsed_response

    def create_ticket(self, category: str, details: Dict) -> Dict:
        ticket = {
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "status": "open",
            **details
        }
        return ticket

# Example usage
def main():
    assistant = SupportAssistant()
    print("Support Assistant: Hello! How can I help you today?")
    
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break
            
        response = assistant.get_completion(user_input)
        if response.get("end_conversation"):
            print("Support Assistant: Thank you for the information. Your ticket has been created.")
            print("Generated Ticket:", json.dumps(response, indent=2))  # Show ticket details
            break
        
        # If no end_conversation, show assistant message
        print(f"Support Assistant: {response.get('message', response)}")


if __name__ == "__main__":
    main()