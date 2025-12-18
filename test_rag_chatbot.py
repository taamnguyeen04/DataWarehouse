import requests
import json

def test_chat():
    url = "http://localhost:8001/chat"
    
    # Sample query (a simplified version of the patient description)
    query = "The patient is a 34-year-old female, primigravida. At 33 weeks, polyhydramnios and fetal limb abnormalities were noted. The baby was born prematurely at 36 weeks with respiratory failure and hypotonia. Genetic testing revealed TTN variants. What is the diagnosis?"
    
    payload = {
        "query": query,
        "top_k": 5
    }
    
    print(f"Sending query to {url}...")
    print(f"Query: {query}")
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        print("\n" + "="*50)
        print("GEMINI RESPONSE:")
        print("="*50)
        print(data["answer"])
        print("\n" + "-"*50)
        print("References:")
        for ref in data["references"]:
            print(f"- PMID: {ref}")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure 'rag_chatbot_api.py' is running.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_chat()
