# Install necessary packages

%pip install langchain tiktoken requests pandas openai

dbutils.library.restartPython()

import requests
import json
from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, AIMessage
import pandas as pd


# Initialize Azure AI Search Service

azure_search_service_name = AI_SEARCH_SERVICE_NAME
azure_search_index_name = AI_SEARCH_INDEX_NAME
azure_search_api_key = AI_SEARCH_KEY
azure_search_api_version = "2023-11-01"


def retrieve_documents(query):
    # Endpoint URL for the Azure Search service
    search_endpoint = f"https://{azure_search_service_name}.search.windows.net/indexes/{azure_search_index_name}/docs?api-version={azure_search_api_version}&search={query}"

    # Headers for the HTTP request
    headers = {
        "Content-Type": "application/json",
        "api-key": azure_search_api_key
    }

    # HTTP GET request to Azure Search service
    response = requests.get(search_endpoint, headers=headers)

    # Check if the response status is 200 (OK)
    if response.status_code == 200:
        return response.json()["value"]
    else:
        # Print an error message and return an empty list if the request failed
        print("Error retrieving documents:", response.text)
        return []


def generate_prompt(question, documents):
    # Initialize the prompt with the provided question
    prompt = question + "\n\n"

    # Iterate through each document in the provided list
    for doc in documents:
        prompt += f"Document: {doc['content']}\n\n"
    prompt += "Answer:"

    return prompt


# Initialize Azure OpenAI Service and create an instance

openai_api_type = "azure"
openai_api_base = OPENAI_ENDPOINT
openai_api_version = "2023-07-01-preview" 
deployment_name = "chat"
openai_api_key = OPENAI_API_KEY

client = AzureChatOpenAI(
    azure_deployment=deployment_name,
    azure_endpoint=openai_api_base,
    openai_api_key=openai_api_key,
    openai_api_version=openai_api_version
)


def ask_question(question):
    documents = retrieve_documents(question)
    prompt = generate_prompt(question, documents)

    # Create a HumanMessage object for the user's question
    user_message = HumanMessage(content=prompt)

    # Using AzureChatOpenAI client for the API call
    response = client(messages=[user_message])

    # Parsing the response
    return response.content if isinstance(response, AIMessage) else "No response generated."


prompts = []

# Test Queries for Federal Holidays
prompts.append("What are the federal holidays observedin January 2023?")
prompts.append("If a federal holiday falls on a Sunday, what is the policy for observing it?")
prompts.append("Is Columbus Day a federal holiday in 2024, and if so, on what date is it observed?")

# Test Queries for Bug Reporting Directions
prompts.append("Explain the difference between Expected Behavior and Actual Behavior in bug reporting based on the document.")
prompts.append("What are the guidelines for including screenshots and error messages in bug reports?")
prompts.append("Can you outline the three main categories for prioritizing bugs as described in the document?")
prompts.append("Provide an example of how to handle a high-priority bug according to the Bug Reporting Directions.")

# Test Queries for Sample Benefit Guide
prompts.append("For my company group's insurance plan, if I separate from the company, will my insurance plan continue?")
prompts.append("What are the eligibility requirements for enrolling in the health insurance program?")
prompts.append("Describe the procedure for changing or updating personal insurance information.")
prompts.append("Explain the dental insurance benefits and the covered procedures.")
prompts.append("Detail the vision insurance coverage, including exams and eyewear allowances.")
prompts.append("Outline the steps for filing a claim under the company's health insurance policy.")
prompts.append("Summarize the available options for life insurance coverage and their respective benefits.")
prompts.append("Discuss the Employee Assistance Program and its services.")
prompts.append("Provide information about the retirement savings plan offered, including contribution limits.")
prompts.append("Explain the short-term and long-term disability insurance benefits.")

# Test Queries for Sample Employee Handbook
prompts.append("What is the policy on voluntary at-will employment as described in the Employee Handbook?")
prompts.append("Can you summarize the Equal Employment Opportunity policy in the handbook?")
prompts.append("Describe the policy against workplace harassment outlined in the Employee Handbook.")
prompts.append("What are the guidelines for solicitation in the workplace as per the handbook?")
prompts.append("Explain the hours of work, attendance, and punctuality policies in the Employee Handbook.")
prompts.append("Detail the overtime policy for employees as mentioned in the handbook.")
prompts.append("Outline the various employment categories defined in the Employee Handbook.")
prompts.append("Describe the process of work review and performance evaluation in the handbook.")
prompts.append("Summarize the health and life insurance benefits available to employees as stated in the handbook.")
prompts.append("What are the leave benefits and other work policies mentioned in the Employee Handbook?")

# Test Queries for Sample Training Development Guide
prompts.append("What types of training expenses are covered under the company’s training and development policy?")
prompts.append("Can you outline the approval process for employee-driven training programs?")
prompts.append("Describe the repayment schedule for training programs if an employee leaves the company.")
prompts.append("What criteria are considered for approving off-site training and development courses?")
prompts.append("How is on-site training handled and what are the criteria for employee participation?")


# Model Responses
model_responses = {}

for query in prompts:
    response = ask_question(query)
    model_responses[query] = response


output_table = pd.DataFrame(list(model_responses.items()), columns=['Query', 'AI Model Response'])
display(output_table)



# CODE BELOW IS FOR CONVERSATIONAL AGENT

from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import Tool
from langchain.agents import initialize_agent


# chat completion llm
llm = client

# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5, return_messages=True
)

# Convert retrieval chain into a tool
tools = [
    Tool(
        name="Knowledge Base",
        func=ask_question,
        description=(
            "use this tool when answering general knowledge queries to get "
            "more information about the topic"
        ),
    )
]

# Initialize Agent
agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    memory=conversational_memory,
)


# Testing the agent

query = "Provide an example of how to handle a high-priority bug according to the Bug Reporting Directions."
agent(query)

agent("Can you summarize this?")

agent("What is the next upcoming work holiday?")
