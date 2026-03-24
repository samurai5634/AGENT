import streamlit as st
from crewai import Crew, Process
import agent  # Your agent.py
import tasks  # Your tasks.py

st.set_page_config(page_title="MAS Support System", layout="wide")

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
    status_map[key].write(f" {key}: Waiting")

st.title("Multi-Agent Customer Support System")
st.markdown("---")

# User Input
user_query = st.text_area("Enter Customer Support Ticket:", placeholder="e.g., My payment was deducted but the order status is still 'Failed'...")

col1, col2 = st.columns([1, 5])
submit = col1.button("Analyze Ticket")

if submit and user_query:
    # 1. Update Sidebar to show we've started
    status_map["Summarizer"].write(" Summarizer: Working...")

    # 2. Setup the Crew
    # We use the tasks you already defined in tasks.py
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
        status_map[key].write(f" {key}: Completed")