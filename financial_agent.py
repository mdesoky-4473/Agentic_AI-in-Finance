from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os

Groq.api_key = os.getenv("GROQ_API_KEY")

# create an Agents
# websearch Agent
websearch_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model = Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls= True,
    markdown=True,
)

#financial agent
finance_agent = Agent(
    name="Finance AI Agent",
    model =Groq(id="llama-3.3-70b-versatile"),
    tools= [YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,company_news=True)],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

# Combine 2 Agents
multi_ai_agents = Agent(
    team=[websearch_agent,finance_agent],
    model =Groq(id="llama-3.3-70b-versatile"),
    instructions=["Always include sources","Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agents.print_response("Summarize analyst recommendation and share the latest news for NVDA",stream=True)