from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

# Use a Gemini model provided by Google's generative API.
# Change the model string to the specific Gemini variant you want to target.
llm = ChatGoogleGenerativeAI(model="gemini-1.0")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an agent that reflects on observations and memories."),
    ("human", "Memory: {memory}\nObservation: {observation}\nReflect:")
])

def reflect_with_llm(memory, observation):
    """
    Build the prompt and call the Google Gemini model via the LangChain wrapper.
    This function attempts several common call patterns so it works across
    different LangChain / wrapper versions.
    """
    # Format the prompt text (use a couple of fallbacks for API differences)
    try:
        text = prompt.format(memory=str(memory), observation=observation)
    except Exception:
        try:
            prompt_obj = prompt.format_prompt(memory=str(memory), observation=observation)
            text = str(prompt_obj)
        except Exception:
            text = f"Memory: {memory}\nObservation: {observation}\nReflect:"

    # Try calling the model in several ways (callable, generate, invoke)
    try:
        # Many LangChain chat models are callable and return a string
        result = llm(text)
        return result
    except Exception:
        pass

    try:
        # Some versions expose a 'generate' method returning an LLMResult
        llm_result = llm.generate([text])
        if hasattr(llm_result, "generations"):
            gens = llm_result.generations
            if gens and gens[0] and hasattr(gens[0][0], "text"):
                return gens[0][0].text
        return str(llm_result)
    except Exception:
        pass

    try:
        # older wrappers sometimes expose an 'invoke' method
        invoked = llm.invoke(text)
        if hasattr(invoked, "content"):
            return invoked.content
        return str(invoked)
    except Exception as e:
        return f"LLM call failed: {e}"
