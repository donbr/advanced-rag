# llm_models.py
from langchain_openai import ChatOpenAI
from src import settings # To ensure API keys are set
import logging

logger = logging.getLogger(__name__)

# settings.setup_env_vars() # Called when settings is imported

def get_chat_model():
    logger.info("Initializing ChatOpenAI model (default: gpt-3.5-turbo)...")
    try:
        model = ChatOpenAI() # Uses default model gpt-3.5-turbo
        logger.info(f"ChatOpenAI model initialized successfully: {model.model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize ChatOpenAI model: {e}", exc_info=True)
        raise # Reraise to signal failure

if __name__ == "__main__":
    if not logging.getLogger().hasHandlers(): # Check if root logger is configured
        if 'logging_config' not in globals(): # Simple check if logging_config was imported
            from src import logging_config # Make sure it's available from src package
        logging_config.setup_logging()

    logger.info("--- Running llm_models.py standalone test ---")
    try:
        chat_model = get_chat_model()
        if chat_model:
            logger.info(f"Chat model type: {type(chat_model)}")
            # Example invocation (optional, can be verbose)
            # from langchain_core.messages import HumanMessage
            # logger.info("Attempting a test invocation...")
            # response = chat_model.invoke([HumanMessage(content="Hello! Give a very short reply.")])
            # logger.info(f"Test invocation response: {response.content}")
        else:
            logger.error("Chat model could not be initialized in standalone test.")
    except Exception as e:
        logger.error(f"Error during llm_models.py standalone test: {e}", exc_info=True)
    logger.info("--- Finished llm_models.py standalone test ---") 