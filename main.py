import streamlit as st
import os
from groq import Groq
from dotenv import load_dotenv
import time # To add a small delay for user experience

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
DEFAULT_MODEL = "llama3-8b-8192" # Or choose another model like "mixtral-8x7b-32768"

# --- Helper Functions ---
def initialize_groq_client():
    """Initializes and returns the Groq client if the API key is valid."""
    # Prioritize environment variables, then Streamlit secrets
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key and "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]

    if not api_key:
        st.error("Groq API key not found. Please set GROQ_API_KEY in your environment variables or Streamlit secrets.")
        st.stop() # Halt execution if no key

    try:
        client = Groq(api_key=api_key)
        # Test connection (optional, but good practice)
        client.models.list()
        return client
    except Exception as e:
        st.error(f"Failed to initialize or connect with Groq client: {e}")
        st.stop()

def get_groq_response(client, messages, model=DEFAULT_MODEL):
    """Gets a response from the Groq API."""
    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.7,
            max_tokens=1500, # Increased slightly for potentially more detailed answers
            top_p=1,
            stop=None,
            stream=False,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred while communicating with Groq: {e}")
        return None

# --- Function to build the System Prompt ---
def create_system_prompt(child_age_range, temperament_traits, current_challenges):
    """Creates the system prompt incorporating the child's details."""
    prompt_lines = [
        "You are a helpful, empathetic, and knowledgeable AI assistant specializing in providing personalized parenting tips and advice.",
        "Your tone should be supportive, understanding, and non-judgmental.",
        "Focus on practical, actionable suggestions and positive reinforcement techniques.",
        "Avoid overly technical jargon unless explaining a specific concept clearly.",
        "Respond concisely but thoroughly to user questions about parenting challenges and strategies."
    ]

    # Add context based on form input
    if child_age_range and child_age_range != "Not specified":
         prompt_lines.append(f"The user is specifically asking for advice related to a child in the {child_age_range} age range. Tailor your advice significantly based on this developmental stage.")
    else:
        prompt_lines.append("The user has not specified a child's age range, so provide general advice or ask for clarification if age is crucial for the specific question.")

    if temperament_traits: # If the list is not empty
        traits_string = ", ".join(temperament_traits)
        prompt_lines.append(f"The child's temperament is described as: {traits_string}. Keep these traits in mind when suggesting communication styles, activities, and discipline strategies.")

    if current_challenges and current_challenges.strip(): # If the string is not empty or just whitespace
        prompt_lines.append(f"The parent mentioned they are currently focusing on or facing challenges with: '{current_challenges.strip()}'. Try to address this area proactively if relevant to the user's questions, or use it as context for your advice.")

    prompt_lines.append("Always prioritize safety and well-being in your advice. If a topic seems potentially serious (e.g., medical issues, severe behavioral problems), gently suggest consulting a professional (pediatrician, therapist, etc.).")

    return "\n".join(prompt_lines)

# --- Streamlit App ---

# --- Page Configuration ---
st.set_page_config(
    page_title="Personalized Parenting Tips Chatbot",
    page_icon="🧸",
    layout="centered"
)

st.title("🧸 Personalized Parenting Tips Chatbot")

# --- Initialize Session State ---
if "form_completed" not in st.session_state:
    st.session_state.form_completed = False
    st.session_state.child_age = None
    st.session_state.child_temperament = [] # Initialize as empty list
    st.session_state.current_challenges = "" # Initialize as empty string
    st.session_state.messages = []
    st.session_state.client = None

# --- State Management: Show Form or Chat ---

# --- 1. Display Form if not completed ---
if not st.session_state.form_completed:
    st.info("Tell us a bit about your child to receive more personalized advice.")

    with st.form("user_info_form"):
        st.subheader("Child Information")

        # --- Age Range ---
        age_options = [
            "Not specified", "Newborn (0-3 months)", "Infant (3-12 months)",
            "Toddler (1-3 years)", "Preschooler (3-5 years)",
            "School-Age (6-12 years)", "Teenager (13+ years)",
        ]
        selected_age = st.selectbox(
            "Child's age range:",
            options=age_options,
            index=0, # Default to 'Not specified'
            key="form_age" # Add key for potential later access if needed
        )

        # --- Temperament ---
        temperament_options = [
            "Easygoing / Adaptable", "Active / Energetic", "Shy / Cautious",
            "Intense / Sensitive", "Distractible", "Persistent / Strong-willed"
        ]
        selected_temperaments = st.multiselect(
            "Select traits that describe your child's general temperament (optional):",
            options=temperament_options,
            key="form_temperament"
        )

        # --- Current Challenges/Goals ---
        challenges_input = st.text_area(
            "Briefly mention any current challenges or parenting goals you have (optional, e.g., 'managing screen time', 'improving communication', 'dealing with picky eating'):",
            key="form_challenges",
            height=100
        )

        # --- Submit Button ---
        submitted = st.form_submit_button("Start Chatting")

        if submitted:
            # Store form data in session state
            st.session_state.child_age = selected_age
            st.session_state.child_temperament = selected_temperaments
            st.session_state.current_challenges = challenges_input

            # Mark form as completed and rerun
            st.session_state.form_completed = True
            st.success("Thanks! Setting up your personalized chat...")
            time.sleep(1.5) # Slightly longer pause
            st.rerun()


# --- 2. Display Chat Interface if form IS completed ---
else:
    # --- Initialization (only run once after form completion) ---
    if st.session_state.client is None:
        with st.spinner("Connecting to the AI assistant..."):
             st.session_state.client = initialize_groq_client()

        # Initialize chat history *after* form is complete and details are known
        if not st.session_state.messages:
            system_prompt = create_system_prompt(
                st.session_state.child_age,
                st.session_state.child_temperament,
                st.session_state.current_challenges
            )
            st.session_state.messages = [{"role": "system", "content": system_prompt}]

            # Create a more personalized welcome message
            welcome_message_parts = ["Hello! I'm ready to help with parenting tips"]
            if st.session_state.child_age and st.session_state.child_age != "Not specified":
                 welcome_message_parts.append(f"for your {st.session_state.child_age.lower()}")
            if st.session_state.child_temperament:
                 temperament_str = ", ".join(st.session_state.child_temperament).lower()
                 welcome_message_parts.append(f" (described as {temperament_str})")
            welcome_message_parts.append(".")
            if st.session_state.current_challenges and st.session_state.current_challenges.strip():
                 welcome_message_parts.append(f" I see you're interested in '{st.session_state.current_challenges.strip()}'.")
            welcome_message_parts.append(" How can I assist you today?")

            st.session_state.messages.append({"role": "assistant", "content": " ".join(welcome_message_parts)})


    # --- Display Chat Context & History ---
    # Display the context provided by the user
    context_summary = f"Child Age: {st.session_state.child_age}"
    if st.session_state.child_temperament:
        context_summary += f" | Temperament: {', '.join(st.session_state.child_temperament)}"
    if st.session_state.current_challenges and st.session_state.current_challenges.strip():
        context_summary += f" | Focus: {st.session_state.current_challenges.strip()[:50]}..." # Truncate long text

    st.caption(f"Context: {context_summary}")
    st.caption(f"Powered by Groq ({DEFAULT_MODEL})")


    if st.session_state.client:
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # --- Handle User Input ---
        prompt = st.chat_input("Ask for parenting advice...")

        if prompt:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get response
            with st.spinner("Thinking..."):
                messages_for_api = st.session_state.messages
                assistant_response = get_groq_response(
                    st.session_state.client,
                    messages_for_api,
                    model=DEFAULT_MODEL
                )

            # Add assistant response
            if assistant_response:
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                with st.chat_message("assistant"):
                    st.markdown(assistant_response)

    else:
         st.error("Chatbot could not be initialized. Please check API key and connection.")

    # --- Reset Button ---
    if st.sidebar.button("Restart Chat / Edit Child Info"):
         # Reset relevant session state variables
         st.session_state.form_completed = False
         st.session_state.child_age = None
         st.session_state.child_temperament = [] # Reset list
         st.session_state.current_challenges = "" # Reset string
         st.session_state.messages = []
         st.session_state.client = None
         st.rerun()