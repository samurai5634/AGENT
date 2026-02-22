from crewai import Agent,LLM
import tools

llm = LLM(
    model="ollama/llama3.2:latest",
    base_url="http://localhost:11434"
)

triager = Agent(
    role='Support Triager',
    goal='Categorize incoming tickets and assess urgency'
            'Your task is to predict Department, Priority, Sentiment and action-type for a support query',
    backstory='You are an expert at identifying the core issue in customer messages.'
                'You are well equipped with all the knowledge to predict appropriate action type',
    tools=[tools.triage_tool],
    llm=llm
)


researcher = Agent(
    role='Knowledge Specialist',
    goal='Find the best historical solution for the query',
    backstory='You have access to all past resolved tickets and technical docs.',
    tools=[tools.knowledge_base_tool],
    memory = True,
    verbose = True,
    llm=llm
)

summary_specialist = Agent(
    role='Technical Summary Specialist',
    goal='Distill complex customer queries into concise, 10-12 word technical summaries.',
    backstory="""You are an expert technical writer. You excel at removing emotional noise, 
    polite greetings, and repetitive phrases to focus purely on the technical root cause 
    of a support ticket. Your summaries are used by senior engineers to understand 
    problems at a glance.""",
    verbose=True,
    allow_delegation=False,
    llm=llm  # Using your Ollama instance
)

complexity_analyst = Agent(
    role='Complexity Auditor',
    goal='Analyze customer queries to determine technical difficulty on a scale of 1-10.',
    backstory="""You are a Senior Technical Lead. You evaluate how many resources 
    and how much expertise is needed to solve a ticket. You look for technical 
    keywords and the severity of the mentioned issue.""",
    llm=llm, 
    verbose=True,
    reasoning=True # This makes the agent "think" before scoring
)

time_agent = Agent(
    role='SLA and Policy Enforcement Officer',
    goal='Ensure ticket resolution commitments align with business SLAs and technical complexity.',
    backstory="""You are a veteran Service Level Manager. Your expertise lies in 
    identifying high-risk tickets that appear simple but carry hidden technical debt. 
    You review the initial triage and complexity scores to calculate realistic 
    resolution times. If you detect a breach of the 240/540/1080 minute SLA 
    thresholds, you have the authority to override the initial action and 
    escalate the ticket immediately to prevent customer dissatisfaction.""",
    tools=[tools.overriding_tool, tools.estimate_resolution_time],
    verbose=True,
    allow_delegation=False,
    memory=True,
    llm = llm
)


orchestrator_agent = Agent(
    role='Customer Support Orchestrator',
    goal='Synthesize multiple technical insights into a single clear routing decision.',
    backstory="""You are the final decision-maker. You don't perform the 
    technical analysis yourself, but you review the reports from the 
    Triage Specialist, the Auditor, and the Policy Officer. Your job is 
    to create a final, unified ticket brief that is ready for human action.""",
    verbose=True,
    allow_delegation=False,
    llm = llm
)




