from google import genai

key_file_path = "APIs/keys/gemini_key.txt"
with open(key_file_path, 'r') as f:
    api_key = f.read().strip()


client = genai.Client(api_key=api_key)

prompt1 = "The user hes inputed a following text:"
prompt2 = "Find me the yelow mustard bottle."
prompt3 = "I have following object I can choose from: foam brick, mustard bottle, gray box, box of jello and cracker box. What object is the user refering to? Answer only with the name of the object if none of the objects are refered to, answer with 'none'. "

final_prompt = f"{prompt1} {prompt2} {prompt3}"

response = client.models.generate_content(
    model="gemini-2.0-flash", contents=final_prompt
)

print(response.text)
