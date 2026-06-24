import streamlit as st
from crewai import Crew, Process
import agent  # Your agent.py
import tasks  # Your tasks.py
import sqlite3
import pandas as pd
import uuid  # Imported to generate unique ticket IDs for the database

# Import your database functions
from database import init_ticket_db, commit_processed_ticket

st.set_page_config(page_title="MAS Support System", layout="wide")

# Initialize the local relational database on application startup
@st.cache_resource
def setup_database():
    init_ticket_db()

setup_database()

# Sidebar for Status Tracking
st.sidebar.title("Agent Pipeline Status")
status_map = {
    "Summarizer": st.sidebar.empty(),
    "Triager": st.sidebar.empty(),
    "Auditor": st.sidebar.empty(),
    "Researcher": st.sidebar.empty(),
    "Policy/SLA": st.sidebar.empty(),
    "Orchestrator": st.sidebar.empty()
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
    # 1. Update Sidebar to show we've started
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

    # 3. Execution
    with st.status("Agents are collaborating...", expanded=True) as status:
        # CrewAI automatically injects user_query into any task with {query}
        final_report = support_crew.kickoff(inputs={'query': user_query})
        status.update(label="Analysis Complete!", state="complete")

    # 4. Final Display
    st.subheader("Final Orchestrated Brief")
    st.markdown(final_report)

    # Update Sidebar to Finished
    for key in status_map:
        status_map[key].write(f" ✅ {key}: Completed")

    # 5. Database Integration: Construct payload and commit data
    try:
        # Check if the final task or crew returned structured output (Pydantic), 
        # otherwise fallback safely to string extractions or placeholders.
        structured_data = getattr(final_report, "pydantic", None)
        
        ticket_payload = {
            'id': str(uuid.uuid4())[:8],  # Generate a short unique ID for the ticket
            'query': user_query,
            'dept': getattr(structured_data, 'predicted_department', 'Unknown'),
            'priority': getattr(structured_data, 'predicted_priority', 'Normal'),
            'sentiment': getattr(structured_data, 'predicted_sentiment', 'Neutral'),
            'action': getattr(structured_data, 'predicted_action', 'Review Required'),
            'complexity': float(getattr(structured_data, 'complexity_score', 0.0)),
            'time_pred': float(getattr(structured_data, 'predicted_resolution_time', 0.0)),
            'override': getattr(structured_data, 'final_override_status', 'No Override'),
            'faithfulness': str(getattr(structured_data, 'faithfulness_score', 'N/A')),
            'relevance': str(getattr(structured_data, 'relevance_score', 'N/A')),
            'audit_status': getattr(structured_data, 'audit_status', 'Logged')
        }
        
        # Safely insert the telemetry payload into your enterprise_itsm.db
        commit_processed_ticket(ticket_payload)
        st.success(f"Ticket telemetry saved successfully under ID: {ticket_payload['id']}")
        
    except Exception as db_err:
        st.warning(f"Metadata compiled successfully, but skipped automated database payload parsing: {db_err}")