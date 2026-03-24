
from crewai import Agent, Task, Crew,LLM
import os
os.environ["LITELLM_LOGGING"] = "False"

llm1 = LLM(
    model="ollama/llama3.2:latest",
    base_url="http://localhost:11434",
    )


def test_ollama():
    print("--- Testing Connection to Ollama ---")
    
    # 1. Define a simple test agent
    tester = Agent(
        role='System Tester',
        goal='Verify that you can communicate with the user.',
        backstory='You are a diagnostic tool testing a local LLM connection.',
        llm=llm1,
        verbose=True
    )

    # 2. Define a simple test task
    test_task = Task(
        description="what are the best 5 places to visit in pune.",
        expected_output="A confirmation message.",
        human_input=True,
        agent=tester
    )

    # 3. Create a mini crew
    test_crew = Crew(
        agents=[tester],
        tasks=[test_task],
        memory = True,
        verbose = True
    )

    # 4. Kick it off
    try:
        result = test_crew.kickoff()
        print("\n--- TEST RESULT ---")
        print(result)
        print("--------------------")
    except Exception as e:
        print(f"\n[ERROR] Could not connect to Ollama: {e}")
        print("Check if Ollama is running at http://localhostp:11434")


if __name__ == "__main__":
    test_ollama()