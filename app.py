from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading model and tokenizer...")
try:
    import torch
    # Limit PyTorch to 1 thread to minimize background memory and CPU usage
    torch.set_num_threads(1)
    
    # Using distilgpt2 (82M parameters) which is extremely lightweight and RAM-friendly
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True
    )
    logger.info(f"{model_name} loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load the model. Ensure you have enough memory (4GB+). Error: {e}")
    import sys
    sys.exit(1)
app = Flask(__name__)

# Global state to keep chat history
chat_history_ids = None

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
        logger.info(f"Incoming message from user: {msg}")
        response = get_Chat_response(msg)
        logger.info(f"Generated response: {response}")
        return str(response)
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return "Sorry, I encountered an error."

def get_Chat_response(text):
    global chat_history_ids
    try:
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        # limit history to last 100 tokens to drastically reduce memory allocation growth
        if chat_history_ids is not None:
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)[:, -100:]
        else:
            bot_input_ids = new_user_input_ids

        # create attention mask
        attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)

        # generate a response while limiting the total chat history to 256 tokens (lower memory footprint)
        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=256,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask
        )

        # decode the response
        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response

    except Exception as e:
        logger.error(f"Model generation error: {e}")
        return "I'm having trouble thinking right now."

if __name__ == '__main__':
    app.run()

