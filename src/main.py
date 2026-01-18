
from crewai import Crew, Process
from agent import (
    summary_specialist, triager, complexity_analyst, 
    researcher, time_agent, orchestrator_agent
)
from src.tasks import (
    summary_task, triage_task, complexity_task, 
    research_task, override_task,orchestrator_task
)

def run_customer_support_system():
    # 1. Initialize the Crew
    # The order in 'tasks' defines the execution sequence
    support_crew = Crew(
        agents=[
            summary_specialist, 
            triager, 
            complexity_analyst,
            researcher, 
            time_agent, 
            orchestrator_agent
        ],
        tasks=[
            summary_task,      # 1. Headline generation
            triage_task,       # 2. ML Classification
            complexity_task,   # 3. Technical Scoring
            research_task,     # 4. KNN Historical Search
            override_task,     # 5. SLA Governance & Buffering
            orchestrator_task  # 6. Final Briefing
        ],
        process=Process.sequential, # Ensures context flows Task 1 -> Task 2...
        verbose=True
    )

    # 2. Live Demo Loop
    print("=== Agentic Customer Support System: Live Demo ===")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Enter Customer Query: ")
        if query.lower() == 'exit':
            break

        print(f"\n[System] Processing ticket using Agent Architecture...\n")
        
        # Start the execution
        # CrewAI injects the 'query' into all tasks with the {query} placeholder
        result = support_crew.kickoff(inputs={'query': query})

        print("\n" + "="*50)
        print("FINAL TICKET ROUTING REPORT")
        print("="*50)
        print(result)
        print("="*50 + "\n")

if __name__ == "__main__":
    
    run_customer_support_system()