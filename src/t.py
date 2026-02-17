from crewai import Agent, Task, Crew, LLM

llm = LLM(
    model="ollama/llama3",
    base_url="http://localhost:11434"
)

agent = Agent(
    role="Tester",
    goal="Test Ollama connection",
    backstory="QA Engineer",
    llm=llm,
    verbose=True
)

task = Task(
    description="Say hello",
    agent=agent
)

crew = Crew(
    agents=[agent],
    tasks=[task]
)

print(crew.kickoff())
