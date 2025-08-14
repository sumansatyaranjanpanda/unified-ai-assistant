from agents.healthcare_agent import run_healthcare_agent
from agents.agriculture_agent import run_agriculture_agent
from agents.finance_agent import run_finance_agent

def router_to_agent(domain):

    if domain == "Healthcare":
        run_healthcare_agent()
    elif domain == "Agriculture":
        run_agriculture_agent()
    elif domain == "Finance":
        run_finance_agent()
    else:
        raise ValueError("Invalid domain selected")