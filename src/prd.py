# production_pipeline.py
import pandas as pd
import uuid
import chromadb
import re
from dotenv import load_dotenv
from crewai import Crew, Process

# Import your underlying database structures
from database import init_ticket_db, commit_processed_ticket

# Import your worker configurations
from agent import (
    summary_specialist, triager, complexity_analyst, 
    researcher, time_agent, orchestrator_agent, critic_agent
)
from tasks import (
    summary_task, triage_task, complexity_task, 
    research_task, override_task, orchestrator_task, get_audit_task
)

load_dotenv()

def execute_integrated_system():
    # Initialize your SQLite Relational Database Engine
    init_ticket_db()
    
    # Connect to your seeded local ChromaDB Vector database
    chroma_client = chromadb.PersistentClient(path="./local_vector_db")
    collection = chroma_client.get_collection("troubleshooting_manuals")
    
    # Initialize your CrewAI Multi-Agent Worker Queue
    worker_crew = Crew(
        agents=[summary_specialist, triager, complexity_analyst, researcher, time_agent, orchestrator_agent],
        tasks=[summary_task, triage_task, complexity_task, research_task, override_task, orchestrator_task],
        process=Process.sequential,
        verbose=True
    )
    
    # Load your evaluation test split dataset to simulate real customer incoming traffic
    df = pd.read_csv('../datasets/ticket_test_dataset.csv')
    print(f"\n🚀 Integrated pipeline online. Processing {len(df)} testing tickets...\n")
    
    for index, row in df.iterrows():
        raw_query = str(row['Customer Query'])
        ticket_id = str(row['Ticket ID'])
        
        print(f"\nIngesting Ticket {ticket_id}...")
        
        # 1. DYNAMIC VECTOR RAG SEARCH (No hardcoding)
        search_results = collection.query(query_texts=[raw_query], n_results=1)
        retrieved_kb_context = search_results['documents'][0][0]
        
        # 2. RUN WORKER AGENTS LOOP
        # Kickoff returns the final task output (the orchestrator's Pydantic schema)
        worker_crew_output = worker_crew.kickoff(inputs={
            'query': raw_query,
            'kb_context': retrieved_kb_context
        })
        
        # EXTRACTION LAYER: Capture the real machine outputs from the Orchestrator
        final_brief = worker_crew_output.pydantic
        
        # Dynamic variable mapping from your 6 operational agents:
        actual_dept = final_brief.department
        actual_priority = final_brief.priority
        actual_sentiment = final_brief.sentiment
        actual_action = final_brief.action_type
        actual_time_pred = final_brief.estimated_time
        actual_override = final_brief.override_status
        
        print(f"   ✓ Worker Brief Extracted. Dept: {actual_dept}, Action: {actual_action}, Estimated Time: {actual_time_pred}m")
        
        # 3. DYNAMIC SYSTEM QUALITY AUDITOR (CRITIC LAYER FROM t.py)
        # Pass the REAL worker results and the REAL vector context to the Critic Agent
        audit_task = get_audit_task(
            query=raw_query,
            ml_pred=actual_dept,          # Dynamic incoming value
            agent_res=str(worker_crew_output.raw), # Dynamic text resolution summary
            context=retrieved_kb_context  # Dynamic incoming vector string
        )
        
        critic_crew = Crew(agents=[critic_agent], tasks=[audit_task], verbose=False)
        raw_audit_output = critic_crew.kickoff()
        
        # Run your exact regex rules from t.py to extract actual scores from the LLM text
        f_score = re.search(r'faithfulness["\s:]+(\d+)', str(raw_audit_output.raw), re.I)
        r_score = re.search(r'relevance["\s:]+(\d+)', str(raw_audit_output.raw), re.I)

        actual_faithfulness = int(f_score.group(1)) if f_score else "N/A"
        actual_relevance = int(r_score.group(1)) if r_score else "N/A"
        
        # Calculate dynamic status based on incoming data parameters
        if isinstance(actual_faithfulness, int) and actual_faithfulness > 7:
            actual_audit_status = "PASS"
        else:
            actual_audit_status = "FAIL"
            
        print(f"   ✓ Critic complete. Scores -> Faithfulness: {actual_faithfulness}, Relevance: {actual_relevance} [{actual_audit_status}]")

        # 4. ATOMIC TRANSACTION COMMIT TO SQLITE
        # Package your 100% dynamic metrics into the payload envelope
        ticket_payload = {
            'id': ticket_id,
            'query': raw_query,
            'dept': str(actual_dept),
            'priority': str(actual_priority),
            'sentiment': str(actual_sentiment),
            'action': str(actual_action),
            'time_pred': float(actual_time_pred),
            'override': str(actual_override),
            'faithfulness': str(actual_faithfulness),
            'relevance': str(actual_relevance),
            'audit_status': actual_audit_status
        }
        
        # Commit the true live data to your SQL repository database file
        commit_processed_ticket(ticket_payload)
        print("-" * 70)

if __name__ == "__main__":
    execute_integrated_system()