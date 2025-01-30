import streamlit as st
#import anthropic
#from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
import torch

#device = "cpu" # for GPU usage or "cpu" for CPU usage
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

'''
checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
'''

checkpoint = "HuggingFaceTB/SmolVLM-Instruct"
processor = AutoProcessor.from_pretrained(checkpoint)
#quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForVision2Seq.from_pretrained(
    checkpoint,
    torch_dtype=torch.bfloat16,
    #quantization_config=quantization_config,
)


with st.sidebar:
    anthropic_api_key = st.text_input("Anthropic API Key", key="file_qa_api_key", type="password")
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/1_File_Q%26A.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("📝 Image Q&A with VLM")
uploaded_file = st.file_uploader("Upload an image", type=("png", "jpg"))
question = st.text_input(
    "Ask something about the image",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

#if uploaded_file and question and not anthropic_api_key:
#    st.info("Please add your Anthropic API key to continue.")

#if uploaded_file and question and anthropic_api_key:
if uploaded_file and question:
    image = uploaded_file.read().decode()
    '''
    prompt = f"""{anthropic.HUMAN_PROMPT} Here's an article:\n\n<article>
    {article}\n\n</article>\n\n{question}{anthropic.AI_PROMPT}"""

    client = anthropic.Client(api_key=anthropic_api_key)
    response = client.completions.create(
        prompt=prompt,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model="claude-v1",  # "claude-2" for Claude 2 model
        max_tokens_to_sample=100,
    )
    '''
    

    '''
    system_prompt = "You are an AI assistant. Your task is reply to user questions based on the provided context."
    user_prompt = f"Context: {article}. User question: {question}"
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]


    input_text=tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)
    response = tokenizer.decode(outputs[0]) 
    '''

    # Create input messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        },
    ]

    # Prepare inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = inputs.to(DEVICE)

    # Generate outputs
    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    response = generated_texts[0]
    
    st.write("### Answer")
    st.write(response)
