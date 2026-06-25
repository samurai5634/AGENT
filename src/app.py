import streamlit as st
from crewai import Crew, Process
import agent  # Your agent.py
import tasks  # Your tasks.py
import sqlite3
import pandas as pd
import uuid  # Imported to generate unique ticket IDs for the database

# Import your database functions
from database import init_ticket_db, commit_processed_ticket

# Import the critique execution pipeline helper
import critic

st.set_page_config(page_title="MAS Support System", layout="wide")

# Initialize the local relational database on application startup
@st.cache_resource
def setup_database():
    init_ticket_db()

setup_database()

# Sidebar for Status Tracking (Added Critic to match execution pipeline)
st.sidebar.title("Agent Pipeline Status")
status_map = {
    "Summarizer": st.sidebar.empty(),
    "Triager": st.sidebar.empty(),
    "Auditor": st.sidebar.empty(),
    "Researcher": st.sidebar.empty(),
    "Policy/SLA": st.sidebar.empty(),
    "Orchestrator": st.sidebar.empty(),
    "Quality Critic": st.sidebar.empty()  
}

# Initialize sidebar text
for key in status_map:
    status_map[key].write(f" ⚪ {key}: Waiting")

st.title("Multi-Agent Customer Support System")
st.markdown("---")

# User Input
user_query = st.text_area("Enter Customer Support Ticket:", placeholder="e.g., My payment was deducted but the order status is still 'Failed'...")

col1, col2 = st.columns([1, 5])
submit = col1.button("Analyze Ticket")

if submit and user_query:
    # 1. Update Sidebar to show we've started the sequential processing
    status_map["Summarizer"].write(" ⏳ Summarizer: Working...")

    # 2. Setup the Crew
    support_crew = Crew(
        agents=[
            agent.summary_specialist, 
            agent.triager, 
            agent.complexity_analyst, 
            agent.researcher, 
            agent.time_agent, 
            agent.orchestrator_agent
        ],
        tasks=[
            tasks.summary_task, 
            tasks.triage_task, 
            tasks.complexity_task, 
            tasks.research_task, 
            tasks.override_task, 
            tasks.orchestrator_task
        ],
        process=Process.sequential, # Your Hybrid AI Architecture
        verbose=True
    )

    # 3. Core Crew Execution
    with st.status("Agents are collaborating...", expanded=True) as status:
        final_report = support_crew.kickoff(inputs={'query': user_query})
        status.update(label="Crew Orchestration Complete!", state="complete")

    # Update primary crew steps to completed in Sidebar
    for key in ["Summarizer", "Triager", "Auditor", "Researcher", "Policy/SLA", "Orchestrator"]:
        status_map[key].write(f" ✅ {key}: Completed")

    # 4. Neuro-Symbolic Validation Layer (critic.py)
    status_map["Quality Critic"].write(" ⏳ Quality Critic: Auditing Resolution...")
    
    # Extract structured data from Pydantic output to feed into the Critic model
    structured_data = getattr(final_report, "pydantic", None)
    
    predicted_dept = getattr(structured_data, 'predicted_department', 'Unknown')
    proposed_res = getattr(final_report, 'raw', str(final_report))
    # Safely extracting internal context or using fallback snippet for evaluation grounding
    reference_context = getattr(structured_data, 'extracted_context', 'Internal Reference Manual/KB Policy')

    with st.spinner("System Quality Auditor verifying faithfulness & relevance..."):
        audit_results = critic.execute_live_audit(
            query=user_query,
            ml_pred=predicted_dept,
            agent_res=proposed_res,
            context=reference_context
        )
    
    status_map["Quality Critic"].write(" ✅ Quality Critic: Audit Completed")

    # 5. Final UI Display
    st.subheader("Final Orchestrated Brief")
    st.markdown(final_report)

    # Display real-time critique metrics in an expandable UI component
    with st.expander("🛡️ System Quality Audit Telemetry", expanded=True):
        m1, m2, m3 = st.columns(3)
        m1.metric("Faithfulness Score", f"{audit_results.get('faithfulness', 0)}/10")
        m2.metric("Relevance Score", f"{audit_results.get('relevance', 0)}/10")
        m3.metric("Department Compliance", audit_results.get('compliance', 'N/A'))
        st.caption(f"**Auditor Reasoning:** {audit_results.get('reasoning', 'No rationale provided.')}")

    # 6. Database Integration: Construct payload with live critic data and commit
    try:
        ticket_payload = {
            'id': str(uuid.uuid4())[:8],  # Generate a short unique ID for the ticket
            'query': user_query,
            'dept': predicted_dept,
            'priority': getattr(structured_data, 'predicted_priority', 'Normal'),
            'sentiment': getattr(structured_data, 'predicted_sentiment', 'Neutral'),
            'action': getattr(structured_data, 'predicted_action', 'Review Required'),
            'complexity': float(getattr(structured_data, 'complexity_score', 0.0)),
            'time_pred': float(getattr(structured_data, 'predicted_resolution_time', 0.0)),
            'override': getattr(structured_data, 'final_override_status', 'No Override'),
            'faithfulness': str(audit_results.get('faithfulness', 'N/A')),
            'relevance': str(audit_results.get('relevance', 'N/A')),
            'audit_status': f"Audited - Compliance: {audit_results.get('compliance', 'Logged')}"
        }
        
        # Safely insert the telemetry payload into your enterprise_itsm.db
        commit_processed_ticket(ticket_payload)
        st.success(f"Ticket telemetry saved successfully under ID: {ticket_payload['id']}")
        
    except Exception as db_err:
        st.warning(f"Metadata compiled successfully, but skipped automated database payload parsing: {db_err}")