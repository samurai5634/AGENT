import pandas as pd
import requests
import json
from tqdm import tqdm


# rubric for compexity_score 
RUBRIC = (
    "1 to 3 (Low): Routine request (e.g., password reset, hours of operation). "
    "4 to 7 (Medium): Requires investigation (e.g., internet slow, billing discrepancy). "
    "8 to 10 (High): Critical/Infrastructure failure (e.g. server down, data breach)."
)

def agent_generation_call(query, resolution_time, priority):
    """A high-speed version of your Complexity Auditor for bulk data."""
    # COHERENCE CHECK: If it's a breach (>240m for High), tell Ollama to rewrite
    if priority == 'High' and resolution_time > 240:
        rewrite_prompt = f"Rewrite this query as a critical system emergency. Max 15 words. Original: {query}"
        query = call_ollama(rewrite_prompt) or query

    # SCORING: Get the numeric score using your rubric
    score_prompt = f"Analyze: '{query}'. Provide a complexity score 1-10 based on this: {RUBRIC}. Return ONLY the number."
    raw_score = call_ollama(score_prompt)
    
    try:
        clean_score = float(''.join(c for c in raw_score if c.isdigit() or c == '.'))
    except:
        clean_score = 5.0
        
    return query, clean_score

def call_ollama(prompt):
    try:
        r = requests.post("http://localhost:11434/api/generate", 
                          json={"model": "llama3", "prompt": prompt, "stream": False}, timeout=15)
        return json.loads(r.text)['response'].strip()
    except: return None

# Load and Process
data = pd.read_csv('balanced.csv')
data['Complexity_Score'] = 0.0

print("Generating Intelligent Features...")

for i in tqdm(range(len(data))):
    new_q, score = agent_generation_call(data.at[i, 'Customer Query'], 
                                         data.at[i, 'Resolution_Time_Actual'], 
                                         data.at[i, 'Priority'])
    data.at[i, 'Customer Query'] = new_q
    data.at[i, 'Complexity_Score'] = score
    
    if i % 100 == 0: data.to_csv('finaltraining_data.csv', index=False)

data.to_csv('finaltraining_data.csv', index=False)

print("Done! You now have the training data for your agents.")

