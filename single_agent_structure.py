# %%
### createed 14.3.2025

# %% [markdown]
# This script simulates a political persuasion experiment using TinyTroupe, replicating Amsalem (2019). The agent is pre-defined with demographic traits (e.g., party ID, education, issue salience, political interest) and exposed to a speech by a political candidate on immigration. The agent listens to the speech without acting, ensuring internal processing before recall and persuasion measures. The experiment then assesses the agent’s recall, agreement, perceived persuasiveness, and manipulation checks (differentiation, integration, and ideology perception). Additionally, a political knowledge test is administered to validate prior knowledge. Instead of logging responses throughout the experiment, the full conversation is extracted retroactively using ResultsReducer and stored in a structured DataFrame. This design enables systematic analysis of persuasion effects in simulated agents while maintaining methodological consistency with prior research.

# %%
import numpy as np
import pandas as pd
import os
import json
import tinytroupe
from tinytroupe.agent import TinyPerson
from tinytroupe.factory import TinyPersonFactory
from dotenv import load_dotenv
from tinytroupe.extraction import ResultsReducer
import re

# -----------------------------
# 1️⃣ Load API Key & Setup
# -----------------------------
env_path = "/Users/dromar/tinytroupe/.env.local"
load_dotenv(env_path)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ ERROR: OpenAI API Key is missing! Make sure .env.local is set.")
    exit(1)
print("✅ OpenAI API Key successfully loaded!")

# %%
# -----------------------------
# 2️⃣ Define Agent's Demographics
# -----------------------------
context_text = """
You are a 30-year-old Democratic woman with a Bachelor's degree or higher.
You have medium issue salience, low political knowledge, medium political interest, and a high need to evaluate.
"""

persona_details = {
    "party_affiliation": "Democrat",
    "age": 30,
    "education": "Bachelor or higher",
    "issue_salience": 3,  # Medium (scale 1-5)
    "political_interest": 5,  # Medium (scale 0-10)
    "need_to_evaluate": 5,  # High (scale 1-5)
    "political_knowledge": "low",  # Low knowledge
}

# Initialize the agent
factory = TinyPersonFactory(context_text=context_text)
agent = factory.generate_person(agent_particularities=json.dumps(persona_details), temperature=0)
agent.name = "Democratic_Agent"
print("✅ Agent successfully created!")

# %%
# -----------------------------
# 3️⃣ Exposure to Candidate Speech
# -----------------------------
issue = "immigration"
speech_text = (
    "Paul Miller, a Democratic candidate running for the U.S. Senate, presented his perspective on immigration for the first time last night. "
    "In a speech at a party convention, Mr. Miller stated the following: \n\n"
    "Solving the problem of illegal immigration to the country is a top priority for me. "
    "I favor allowing illegal immigrants who are otherwise law-abiding a path to full citizenship. "
    "We need to naturalize illegal immigrants because it will make our economy grow. "
    "However, we must be careful: allowing everyone to stay may encourage more illegal immigrants to come here, and we don’t want that. "
    "My plan is to balance these two goals."
)
agent.listen(speech_text)

# %%
# -----------------------------
# 4️⃣ Persuasion Measures
# -----------------------------
agent.listen_and_act(
    f"In this question, we would like to know how much of Candidate Miller's position you can remember. "
    f"Please write down, in your own words, up to six arguments describing Candidate Miller’s position on {issue}. "
    "Each argument you write should be one sentence long."
)

agent.listen_and_act(
    f"Please rate the extent to which you agree with Candidate Miller’s position on {issue}. "
    "1=Strongly disagree, 2=Disagree, 3=Neither agree nor disagree, 4=Agree, 5=Strongly agree."
)

agent.listen_and_act(
    "How persuasive do you find Candidate Miller’s speech? "
    "1=Definitely not persuasive, 2=Not persuasive, 3=Neither persuasive nor unpersuasive, "
    "4=Persuasive, 5=Definitely persuasive."
)


# %%
# -----------------------------
# 5️⃣ Manipulation Checks
# -----------------------------
agent.listen_and_act(
    f"How many perspectives relevant to the issue of {issue} does Candidate Miller consider in his speech? "
    "1=One perspective, 2=Two perspectives, 3=Three or more perspectives, 4=Don’t know."
)

agent.listen_and_act(
    f"Does Candidate Miller refer in his speech to some type of trade-off or compromise between conflicting perspectives on {issue}? "
    "1=Yes, 2=No, 3=Don’t know."
)

agent.listen_and_act(
    f"People sometimes think of politics in terms of being liberal or conservative. "
    f"Using the following scale, where would you place Candidate Miller's view on the issue of {issue}? "
    "1=Very liberal, 2=Liberal, 3=Neither liberal nor conservative, 4=Conservative, 5=Very conservative."
)

# %%
# -----------------------------
# 6️⃣ Political Knowledge Test
# -----------------------------
agent.listen_and_act(
    "Who has the final responsibility to decide if a law is constitutional or not? "
    "1=The president, 2=The Congress, 3=The Supreme Court."
)

agent.listen_and_act(
    "Whose responsibility is it to nominate judges to the federal courts? "
    "1=The president, 2=The Congress, 3=The Supreme Court."
)

agent.listen_and_act(
    "What is the main duty of the U.S. Congress? "
    "1=To Write Legislation, 2=To administer the president’s policies, 3=To supervise the states’ governments."
)

# %%
# -----------------------------
# 7️⃣ Extract Conversation History
# -----------------------------
reducer = ResultsReducer()

def extract_interaction(focus_agent, source_agent, target_agent, kind, event, content, timestamp):
    if event == "TALK":
        author = focus_agent.name
    elif event == "CONVERSATION":
        author = "USER" if source_agent is None else source_agent.name
    else:
        return None
    return {"author": author, "message": content, "timestamp": timestamp}

reducer.add_reduction_rule("TALK", extract_interaction)
reducer.add_reduction_rule("CONVERSATION", extract_interaction)

# Convert interaction history to a DataFrame
df = reducer.reduce_agent_to_dataframe(agent, column_names=["author", "message", "timestamp"])


# %%
df

# %%
##### Save as csv

# df.to_csv("f{agentid}_conversation_history.csv", index=False) ### save with id number so we can systematically extract from multiagent simulations


