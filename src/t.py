from crewai import Agent, Task, Crew, Process

# 1. Define the Critic Agent
critic_agent = Agent(
    role='System Quality Auditor',
    goal='Ensure the hybrid support system provides faithful, accurate, and SLA-compliant resolutions.',
    backstory="""You are an expert in Neuro-Symbolic AI evaluation. Your job is to audit 
    the interaction between statistical ML outputs and LLM-generated solutions. 
    You prevent hallucinations by ensuring every claim is grounded in the 
    retrieved technical documentation.""",
    verbose=True,
    allow_delegation=False,
    memory=True
)

# 2. Define the Evaluation Task
evaluation_task = Task(
    description="""
    Perform a multi-dimensional audit of the following ticket resolution:
    
    1. **Faithfulness**: Is the solution derived ONLY from the retrieved context? 
       Identify any information not present in the source documentation.
    2. **Answer Relevance**: Does the solution directly address the user's query 
       and the Priority ({priority}) assigned by the ML model?
    3. **SLA Alignment**: Does the proposed resolution meet the predicted 
       Resolution Time ({predicted_time})?
    
    **Ticket Context**: {retrieved_context}
    **Agent Resolution**: {agent_resolution}
    """,
    expected_output="""A structured evaluation report with a score (0-10) for Faithfulness, 
    Relevance, and a final 'Pass/Fail' for the Neuro-Symbolic override.""",
    agent=critic_agent
)

# 3. Form the Crew
quality_crew = Crew(
    agents=[critic_agent],
    tasks=[evaluation_task],
    process=Process.sequential
)

# 4. Example Execution for a single ticket
result = quality_crew.kickoff(inputs={
    'priority': 'High',
    'predicted_time': '45 mins',
    'retrieved_context': 'Documentation: Server 404 errors are caused by DNS misconfiguration.',
    'agent_resolution': 'The server is down because of a hardware failure. I recommend a full replacement.'
})

print(result)