from ollamaset import local_llm
from crewai import Agent, Task, Crew

def test_ollama():
    print("--- Testing Connection to Ollama ---")
    
    # 1. Define a simple test agent
    tester = Agent(
        role='System Tester',
        goal='Verify that you can communicate with the user.',
        backstory='You are a diagnostic tool testing a local LLM connection.',
        llm=local_llm,
        verbose=True
    )

    # 2. Define a simple test task
    test_task = Task(
        description="Say 'Hello Govind, your local Ollama connection is successful!' and tell me which model you are using.",
        expected_output="A confirmation message.",
        agent=tester
    )

    # 3. Create a mini crew
    test_crew = Crew(
        agents=[tester],
        tasks=[test_task]
    )

    # 4. Kick it off
    try:
        result = test_crew.kickoff()
        print("\n--- TEST RESULT ---")
        print(result)
        print("--------------------")
    except Exception as e:
        print(f"\n[ERROR] Could not connect to Ollama: {e}")
        print("Check if Ollama is running at http://localhost:11434")

if __name__ == "__main__":
    test_ollama()