from database import commit_processed_ticket,init_ticket_db
import pandas as pd

# 1. Initialize the database schema once before running
init_ticket_db()

# 2. Load your test dataset split
df = pd.read_csv('../datasets/ticket_test_dataset.csv')

for index, row in df.iterrows():
    # --- Execute your existing Scikit-Learn/CrewAI workflows here ---
    # mock values below simulating your current pipeline outputs:
    simulated_payload = {
        'id': str(row['Ticket ID']),
        'query': str(row['Customer Query']),
        'dept': 'Hardware Support',        # Output of Triage Agent
        'priority': 'HIGH',                # Output of Triage Agent
        'sentiment': 'NEUTRAL',            # Output of Triage Agent
        'action': 'ESCALATE',              # Output of Action Agent
        'time_pred': 120.5,                # Output of SLA Model Agent (in mins)
        'override': 'ESCALATE_OVERRIDE'    # Applied by your Orchestrator Logic
    }
    
    # 3. Save to database instead of outputting to a CSV
    commit_processed_ticket(simulated_payload)