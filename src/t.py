import pandas as pd
import json
import re
from crewai import Agent, Task, Crew, Process, LLM

llm = LLM(
    model="ollama/llama3.2:latest",
    base_url="http://localhost:11434"
)

# 1. SETUP THE JUDGE (LLM-as-a-Judge)
critic_agent = Agent(
    role='System Quality Auditor',
    goal='Objectively score the faithfulness and relevance of support resolutions.',
    backstory="""You are a senior auditor specialized in Neuro-Symbolic AI. 
    You analyze if an AI's response is grounded in provided evidence (Symbolic) 
    and addresses the user's intent (Neural).""",
    verbose=False,
    allow_delegation=False,
    llm = llm

)

def get_audit_task(query, ml_pred, agent_res, context):
    return Task(
        description=f"""
        AUDIT DATA:
        - User Query: {query}
        - ML Predicted Dept: {ml_pred}
        - Agent Proposed Resolution: {agent_res}
        - Retrieved Reference Context: {context}

        EVALUATION STEPS:
        1. Faithfulness: Is the agent resolution supported by the Reference Context? (Score 0-10)
        2. Relevance: Does the resolution solve the User Query? (Score 0-10)
        3. Compliance: Does the resolution align with the ML Predicted Dept? (Yes/No)
        """,
        expected_output="JSON with keys: 'faithfulness', 'relevance', 'compliance', 'reasoning'",
        agent=critic_agent
    )

# 2. LOAD DATA
df = pd.read_csv('datasets\ticket_test_dataset.csv')
audit_results = []

print(f"Starting audit for {len(df)} tickets...\n")

# 3. EXECUTION LOOP (Iterate through dataset)
# Note: For your project, you might start with a subset like df.head(20)
for index, row in df.iterrows():
    # Simulate the context retrieval (In live system, this comes from your KNN/VectorDB)
    mock_context = f"Internal Knowledge Base: Related to {row['Assigned Department']} protocols."
    
    # Define the Task for this specific ticket
    task = get_audit_task(
        query=row['Customer Query'],
        ml_pred=row['Assigned Department'], # ML Baseline
        agent_res=row['Resolution_Steps'],    # Current Agent Output
        context=mock_context
    )
    
    crew = Crew(agents=[critic_agent], tasks=[task])
    raw_output = crew.kickoff()
    
    # Parse scores from the LLM output (regex to find numbers)
    f_score = re.search(r'faithfulness["\s:]+(\d+)', str(raw_output), re.I)
    r_score = re.search(r'relevance["\s:]+(\d+)', str(raw_output), re.I)
    
    audit_results.append({
        'Ticket ID': row['Ticket ID'],
        'ML_Dept': row['Assigned Department'],
        'Faithfulness': f_score.group(1) if f_score else "N/A",
        'Relevance': r_score.group(1) if r_score else "N/A",
        'Status': "PASS" if (f_score and int(f_score.group(1)) > 7) else "FAIL"
    })
    
    print(f"Processed {row['Ticket ID']}: Status {audit_results[-1]['Status']}")

# 4. SAVE AND DISPLAY TABLE
audit_df = pd.DataFrame(audit_results)
audit_df.to_csv('critic_audit_results.csv', index=False)

print("\n--- CRITIC AUDIT TABLE ---")
print(audit_df.to_string(index=False))