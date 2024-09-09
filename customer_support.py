import streamlit as st
import pandas as pd
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
from dotenv import load_dotenv
import requests
import time
import json
import re

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set up the model
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {
        "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
        "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
    {
        "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
    {
        "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
    {
        "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-exp-0801",
    generation_config=generation_config,
    safety_settings=safety_settings,
)


# Load and preprocess the dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\\Users\\Dell\\Downloads\\wifi_solution.csv")
    return df.to_dict("records")


troubleshooting_data = load_data()

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "mobile_number" not in st.session_state:
    st.session_state.mobile_number = None

if "app_started" not in st.session_state:
    st.session_state.app_started = False


# Function to validate mobile number
def is_valid_mobile_number(number):
    return re.match(r"^\d+$", number) is not None


# Function to start the app
def start_app():
    mobile_number = st.text_input("Please enter your mobile number to start:")
    if st.button("Start"):
        if is_valid_mobile_number(mobile_number):
            st.session_state.mobile_number = mobile_number
            st.session_state.app_started = True
            st.rerun()
        else:
            st.error("Please enter a valid mobile number (numbers only).")


# Main app logic
if not st.session_state.app_started:
    start_app()
else:
    st.write(f"Welcome! Your mobile number: {st.session_state.mobile_number}")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    def call_isp_api(mobile_number):
        api_url = "http://172.16.2.172:8089/dtv/genai/getServiceInfo"
        headers = {"Content-Type": "application/json"}
        payload = {"phoneNumber": str(mobile_number)}
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            st.error(f"An error occurred while calling the API: {str(e)}")
            return {"error": str(e)}

    def generate_response(prompt, chat_history, solution_count):
        context = "\n".join(
            [
                f"{item['Issue']}: {item['Symptoms']} - {item['Possible_Causes']}\n"
                f"Solution 1: {item['Solution_1']}\n"
                f"Solution 2: {item['Solution_2']}\n"
                f"Solution 3: {item['Solution_3']}\n"
                f"Solution 4: {item['Solution_4']}"
                for item in troubleshooting_data
            ]
        )
        full_prompt = f"""You are a specialized WiFi troubleshooting assistant. Your sole purpose is to help users resolve WiFi-related issues based on the provided dataset. Use the following information, chat history, and the user's latest query to provide assistance ONLY for WiFi problems.

    Context (WiFi issues and solutions database):
    {context}

    Chat history:
    {chat_history}

    Latest user query: {prompt}

    Instructions:
    1. Determine if the user's query is related to a WiFi issue from the provided dataset. If not, politely inform the user that you can only assist with WiFi-related problems.
    2. If the query is WiFi-related, analyze it in the context of the chat history to understand the full scope of the issue.
    3. Identify the most likely WiFi problem based on the symptoms described and any previous troubleshooting steps mentioned.
    4. Provide a clear, step-by-step solution tailored to the user's specific situation.
    5. Based on the solution_count, provide the appropriate solution:
    - If solution_count is 0 or 1, provide Solution 1 or Solution 2 respectively.
    - If solution_count is 2, use the phrase "Contact Your ISP" to trigger the API call.
    - If solution_count is 3 or 4, provide Solution 3 or Solution 4 respectively.
    6. Use technical terms when necessary, but always explain them in simple language.
    7. After providing each solution, ask the user if the issue is resolved or if they need further assistance.
    8. Maintain a friendly, patient, and encouraging tone throughout your response.

    Remember to consider the chat history and the number of solutions already provided ({solution_count}).

    AI Assistant:
    """

        response = model.generate_content(full_prompt)
        return response.text

    def process_input(prompt):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display response
        with st.chat_message("assistant"):
            chat_history = "\n".join(
                [f"{m['role']}: {m['content']}" for m in st.session_state.messages[:-1]]
            )
            solution_count = sum(
                1 for m in st.session_state.messages if m["role"] == "assistant"
            )

            if solution_count == 2:
                st.markdown("Let me check if there is a service outage in your area...")
                api_response = call_isp_api(st.session_state.mobile_number)
                if "error" not in api_response:
                    if api_response.get("isServiceAffected", False):
                        response = f"It looks like there is an outage in your area. {api_response.get('errorMsg', '')}"
                        st.error(response)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )
                        return
                    else:
                        st.info(
                            "It looks like there is no outage in your area. Let's try something else."
                        )
                        response = generate_response(
                            prompt, chat_history, solution_count
                        )
                else:
                    st.error(
                        "Unable to check for service outages at the moment. Let's continue with other solutions."
                    )
                    response = generate_response(prompt, chat_history, solution_count)
            else:
                response = generate_response(prompt, chat_history, solution_count)

            st.markdown(response)

            # Check if we've exhausted all solutions
            if solution_count >= 4:
                st.warning(
                    "I apologize, but I've provided all available solutions. If the issue persists, please contact your ISP directly for further assistance."
                )

            # Check if the problem is resolved
            if (
                "problem is resolved" in prompt.lower()
                or "issue is resolved" in prompt.lower()
            ):
                st.markdown(
                    "Great! I'm glad we could resolve your WiFi issue. The chat history will now be cleared for a fresh start."
                )
                st.session_state.messages = []
                st.rerun()

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.empty():
            for i in range(5, 0, -1):
                st.write(f"You can ask your next question in {i} seconds...")
                time.sleep(1)
            st.write("You can now ask your next question.")

    # Sidebar with issue categories
    st.sidebar.title("Common WiFi Issues")
    for index, item in enumerate(troubleshooting_data):
        if st.sidebar.button(item["Issue"], key=f"issue_button_{index}"):
            prompt = f"I'm experiencing {item['Issue']}. The symptoms are {item['Symptoms']}."
            process_input(prompt)

    # User input
    if prompt := st.chat_input("What's your WiFi issue?"):
        process_input(prompt)

    # Reset button
    if st.button("Start a New Chat"):
        st.session_state.messages = []
        st.rerun()
