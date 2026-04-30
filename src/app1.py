import streamlit as st
from crewai import Crew, Process
import agent  # Your agent.py
import tasks  # Your tasks.py

st.set_page_config(page_title="MAS Support System", layout="wide")

# --- Sidebar for Status Tracking ---
st.sidebar.title("Agent Pipeline Status")

# We create the placeholders once
status_map = {
    "Summarizer": st.sidebar.empty(),
    "Triager": st.sidebar.empty(),
    "Auditor": st.sidebar.empty(),
    "Researcher": st.sidebar.empty(),
    "Policy/SLA": st.sidebar.empty(),
    "Orchestrator": st.sidebar.empty()
}

# Initial UI State
for key in status_map:
    status_map[key].write(f"⚪ {key}: Waiting")

# --- Callback Function ---
# This function is triggered by CrewAI every time a task completes
def update_sidebar_callback(task_output):
    # Mapping the description to the sidebar key
    # We use a simple keyword check to see which agent finished
    desc = task_output.description.lower()
    
    if "summarize" in desc:
        status_map["Summarizer"].write("✅ Summarizer: Completed")
        status_map["Triager"].write("⏳ Triager: Working...")
    elif "classify" in desc or "triage" in desc:
        status_map["Triager"].write("✅ Triager: Completed")
        status_map["Auditor"].write("⏳ Auditor: Working...")
    elif "complexity" in desc or "audit" in desc:
        status_map["Auditor"].write("✅ Auditor: Completed")
        status_map["Researcher"].write("⏳ Researcher: Working...")
    elif "research" in desc or "knowledge" in desc:
        status_map["Researcher"].write("✅ Researcher: Completed")
        status_map["Policy/SLA"].write("⏳ Policy/SLA: Working...")
    elif "policy" in desc or "override" in desc or "time" in desc:
        status_map["Policy/SLA"].write("✅ Policy/SLA: Completed")
        status_map["Orchestrator"].write("⏳ Orchestrator: Working...")
    elif "orchestrate" in desc or "final" in desc:
        status_map["Orchestrator"].write("✅ Orchestrator: Completed")

# --- Main UI ---
st.title("Multi-Agent Customer Support System")
st.markdown("---")

user_query = st.text_area("Enter Customer Support Ticket:", placeholder="e.g., My payment was deducted but the order status is still 'Failed'...")

col1, col2 = st.columns([1, 5])
submit = col1.button("Analyze Ticket")

if submit and user_query:
    # 1. Attach the callback to each task dynamically
    # This prevents you from having to modify your tasks.py file
    task_list = [
        tasks.summary_task, 
        tasks.triage_task, 
        tasks.complexity_task, 
        tasks.research_task, 
        tasks.override_task, 
        tasks.orchestrator_task
    ]
    
    for task in task_list:
        task.callback = update_sidebar_callback

    # 2. Start the first status
    status_map["Summarizer"].write("⏳ Summarizer: Working...")

    # 3. Setup the Crew
    support_crew = Crew(
        agents=[
            agent.summary_specialist, 
            agent.triager, 
            agent.complexity_analyst, 
            agent.researcher, 
            agent.time_agent, 
            agent.orchestrator_agent
        ],
        tasks=task_list,
        process=Process.sequential,
        verbose=True
    )

    # 4. Execution
    with st.status("Agents are collaborating...", expanded=True) as status:
        # CrewAI injects user_query into tasks containing {query}
        final_report = support_crew.kickoff(inputs={'query': user_query})
        status.update(label="Analysis Complete!", state="complete")

    # 5. Final Display
    st.subheader("Final Orchestrated Brief")
    st.markdown(final_report)