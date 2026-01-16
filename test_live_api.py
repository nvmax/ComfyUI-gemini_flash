import sys
import os

# Add the nodes folder to path
current_dir = os.path.dirname(os.path.abspath(__file__))
nodes_path = os.path.join(current_dir, 'nodes')
sys.path.append(nodes_path)

print(f"Current Dir: {current_dir}")
print(f"Nodes Path: {nodes_path}")
print(f"File exists: {os.path.exists(os.path.join(nodes_path, 'Gemini_Flash_Node.py'))}")
print(f"Sys Path: {sys.path}")

try:
    import Gemini_Flash_Node
    from Gemini_Flash_Node import GeminiFlash
    import google.genai as genai
    
    print("Imports successful.")
    
    # User provided key
    API_KEY = "AIzaSyBrXxtITNEhnrnGL7mnRYQpCGHJEET_vS4"
    
    print(f"Initializing GeminiFlash with key ending in ...{API_KEY[-4:]}")
    node = GeminiFlash(api_key=API_KEY)
    
    if not node.client:
        print("Error: Client not initialized.")
        sys.exit(1)
        
    print("Client initialized. Sending test request...")
    
    # Test generation
    prompt = "Hello! If you allow this message, reply with 'System Operational'."
    # We call generate_content. Note: The input signature might require specific types if checks are strict, 
    # but based on my refactor it takes simple strings for prompt.
    # The method signature: generate_content(self, prompt, input_type, ...)
    # Defaults: input_type="text"
    
    response, image = node.generate_content(prompt=prompt, input_type="text")
    
    print("-" * 20)
    print("Response from Gemini:")
    print(response)
    print("-" * 20)
    
    if "System Operational" in response:
        print("SUCCESS: Live API test passed.")
    else:
        print("WARNING: Response received but didn't contain expected text. Please check output.")

except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
