import pandas as pd
import json
import re
import os
from crewai import Agent, Task, Crew, LLM

# ---------------------------
# 1. LLM SETUP
# ---------------------------
llm = LLM(
    model="ollama/llama3.2:latest",
    base_url="http://localhost:11434",
    timeout=120
)

# ---------------------------
# 2. AGENT SETUP
# ---------------------------
critic_agent = Agent(
    role='System Quality Auditor',
    goal='Objectively score the faithfulness and relevance of support resolutions.',
    backstory="""You are a senior auditor specialized in Neuro-Symbolic AI. 
    You analyze if an AI's response is grounded in provided evidence (Symbolic) 
    and addresses the user's intent (Neural).""",
    verbose=False,
    allow_delegation=False,
    llm=llm
)

# ---------------------------
# 3. TASK BUILDER
# ---------------------------
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

# ---------------------------
# 4. SAFE LLM CALL (RETRY)
# ---------------------------
def safe_kickoff(crew, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = crew.kickoff()

            if result is None or str(result).strip() == "":
                raise ValueError("Empty LLM response")

            return result

        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed: {e}")

    return None

# ---------------------------
# 5. LOAD DATA
# ---------------------------
df = pd.read_csv(r'../datasets/ticket_test_dataset.csv')

output_file = 'critic_audit_results.csv'

# Resume logic
if os.path.exists(output_file):
    existing_df = pd.read_csv(output_file)
    processed_ids = set(existing_df['Ticket ID'])
    audit_results = existing_df.to_dict('records')
    print(f"🔁 Resuming... {len(processed_ids)} tickets already processed.\n")
else:
    processed_ids = set()
    audit_results = []
    print(f"🚀 Starting fresh for {len(df)} tickets...\n")

# ---------------------------
# 6. MAIN LOOP
# ---------------------------
for index, row in df.iterrows():

    ticket_id = row['Ticket ID']

    # Skip already processed
    if ticket_id in processed_ids:
        continue

    try:
        mock_context = f"Internal Knowledge Base: Related to {row['Assigned Department']} protocols."

        task = get_audit_task(
            query=row['Customer Query'],
            ml_pred=row['Assigned Department'],
            agent_res=row['Resolution_Steps'],
            context=mock_context
        )

        crew = Crew(agents=[critic_agent], tasks=[task])

        raw_output = safe_kickoff(crew)

        if raw_output is None:
            raise ValueError("LLM failed after retries")

        # ---------------------------
        # PARSING OUTPUT
        # ---------------------------
        f_score = re.search(r'faithfulness["\s:]+(\d+)', str(raw_output), re.I)
        r_score = re.search(r'relevance["\s:]+(\d+)', str(raw_output), re.I)

        faithfulness = int(f_score.group(1)) if f_score else None
        relevance = int(r_score.group(1)) if r_score else None

        status = "PASS" if (faithfulness is not None and faithfulness > 7) else "FAIL"

        audit_results.append({
            'Ticket ID': ticket_id,
            'ML_Dept': row['Assigned Department'],
            'Faithfulness': faithfulness if faithfulness is not None else "N/A",
            'Relevance': relevance if relevance is not None else "N/A",
            'Status': status
        })

        print(f"✅ Processed Ticket {ticket_id} → {status}")

    except Exception as e:
        print(f"❌ Error processing Ticket {ticket_id}: {e}")

        audit_results.append({
            'Ticket ID': ticket_id,
            'ML_Dept': row['Assigned Department'],
            'Faithfulness': "ERROR",
            'Relevance': "ERROR",
            'Status': "FAILED"
        })

    # ---------------------------
    # SAVE AFTER EACH TICKET
    # ---------------------------
    pd.DataFrame(audit_results).to_csv(output_file, index=False)

# ---------------------------
# 7. FINAL OUTPUT
# ---------------------------
audit_df = pd.DataFrame(audit_results)

print("\n--- CRITIC AUDIT TABLE ---")
print(audit_df.to_string(index=False))