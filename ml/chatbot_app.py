import streamlit as st
import database
# Import the client and the singleton function (ensures correct loading)
from ai.gemini_client import get_gemini_client, GeminiClient 

# Page Config
st.set_page_config(page_title="MediCure Chat", page_icon="💊")

# --- INITIALIZATION ---

# Initialize Gemini Client as a Singleton and cache it
@st.cache_resource
def setup_gemini_client():
    # Use the singleton function from ai/gemini_client.py
    return get_gemini_client()

# Store the client in the session state for easy access
if "gemini_client" not in st.session_state:
    st.session_state.gemini_client = setup_gemini_client()


# 1. Get Query Parameters (The Link Logic)
query_params = st.query_params
scan_id = query_params.get("scan_id", None)
user_id = query_params.get("user_id", "guest_user")

# Get previous scan_id from session state for context tracking
last_scan_id = st.session_state.get("last_scan_id", None)


# 2. Load Context (The Medicine Data)
context_text = ""
medicine_name = "Unknown Medicine"
current_chat_context = None

if scan_id:
    current_chat_context = database.get_scan_context(scan_id)
    
    if current_chat_context:
        medicine_name = current_chat_context.get('medicine_name', 'Unknown Medicine')
        st.success(f"Connected to scan: {medicine_name}")
        
        # This is the detailed AI context used for prompting
        context_text = (
            f"Context: The user is currently discussing the medicine: {medicine_name} "
            f"with these known details: {str(current_chat_context)}. "
            f"Be concise, informative, and always base your answer on the provided context."
        )
    else:
        st.warning("Scan ID found but medicine data not in database. Chatting generally.")


# 3. Initialize Chat History & New Context Marker 🚀

if "messages" not in st.session_state:
    # First time loading, retrieve history from DB
    history = database.get_chat_history(user_id)
    st.session_state.messages = history
    
    # Clean up any residual system message from old runs before starting
    if st.session_state.messages and st.session_state.messages[0].get("role") == "system":
        del st.session_state.messages[0]


# --- CORE FEATURE: Insert Context Marker ---
is_new_scan = (scan_id != last_scan_id) and scan_id is not None

if is_new_scan and scan_id is not None:
    # 📌 Insert a clear visual divider/header to signal the context change
    new_context_message = {
        "role": "assistant", 
        "content": f"""
        ---
        ## 💊 New Medicine Context: **{medicine_name}**
        All subsequent chat messages will refer to **{medicine_name}**.
        ---
        """
    }
    st.session_state.messages.append(new_context_message)


# 4. Manage System Context (Always at the start of the list)

# Remove old system context if it exists at the start
if st.session_state.messages and st.session_state.messages[0].get("role") == "system":
    del st.session_state.messages[0]

# Insert the LATEST system context at the very beginning (index 0)
if context_text:
    st.session_state.messages.insert(0, {"role": "system", "content": context_text})

# Store the current scan_id for next time
st.session_state.last_scan_id = scan_id


# 5. Display Chat Interface
st.title("MediCure AI Assistant")

# Display previous messages (including the new context marker)
for message in st.session_state.messages:
    if message["role"] != "system": # Don't show system context to user
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# 6. Handle User Input
if prompt := st.chat_input(f"Ask about {medicine_name}..."):
    
    # 6a. Show user message and append to session history
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    database.save_chat_message(user_id, "user", prompt) # Save to DB

    # 6b. Build the full prompt string for the AI
    # This combines the system context and all chat history into one string.
    # The AI reads the full history including the hidden system context at index 0
    full_prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" 
                             for msg in st.session_state.messages])

    # 6c. Call the AI
    try:
        client = st.session_state.gemini_client
        
        with st.spinner("Thinking..."):
            # Pass the single aggregated prompt string to the fixed generate_response method
            response_text = client.generate_response(full_prompt)
        
    except Exception as e:
        st.error("There was an error connecting to the AI. Please check your API key and logs.")
        response_text = f"An internal error occurred: {e}" 

    # 6d. Show AI response
    with st.chat_message("assistant"):
        st.markdown(response_text)
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    database.save_chat_message(user_id, "assistant", response_text) # Save to DB