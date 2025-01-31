import io
import os

import streamlit as st
import torch
from byaldi import RAGMultiModalModel
from PIL import Image
from transformers import (AutoModelForVision2Seq, AutoProcessor,
                          BitsAndBytesConfig)
from transformers.image_utils import load_image
from pdf2image import convert_from_bytes

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

'''
@st.cache_resource  # Streamlit Caching decorator
def load_model_embedding():
    checkpoint = "vidore/colqwen2-v0.1"
    # checkpoint = "vidore/colsmolvlm-alpha"
    docs_retrieval_model = RAGMultiModalModel.from_pretrained(checkpoint)
model_embedding = load_model_embedding()

@st.cache_resource  # Streamlit Caching decorator
def load_model_vlm():
    checkpoint = "HuggingFaceTB/SmolVLM-Instruct"
    processor = AutoProcessor.from_pretrained(checkpoint)
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint,
        #torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )
    return model, processor
model_vlm, processor_vlm = load_model_vlm()
'''


def save_images_to_local(dataset, output_folder="data/"):
    os.makedirs(output_folder, exist_ok=True)

    for image_id, image in enumerate(dataset):
        #image = image_data["image"]

        #if isinstance(image, str):
        #    image = Image.open(image)

        output_path = os.path.join(output_folder, f"image_{image_id}.png")
        image.save(output_path, format="PNG")
        #print(f"Image saved in: {output_path}")





# Home page UI
with st.sidebar:
    "[Source Code](https://huggingface.co/spaces/deepakkarkala/multimodal-rag/tree/main)"

st.title("📝 Image Q&A with VLM")
uploaded_pdf = st.file_uploader("Upload PDF file", type=("pdf"))
query = st.text_input(
    "Ask something about the image",
    placeholder="Can you describe me the image ?",
    disabled=not uploaded_pdf,
)

if uploaded_pdf:
    images = convert_from_bytes(uploaded_pdf.getvalue())
    save_images_to_local(images)
    # index documents using the document retrieval model
    # model_embedding.index(
    #   input_path="data/", index_name="image_index", store_collection_with_index=False, overwrite=True
    #)



if uploaded_pdf and query:
    #image_bytes = uploaded_file.read()
    #image = Image.open(io.BytesIO(image_bytes))


    docs_retrieved = model_embedding.search(query, k=1)
    image_similar_to_query = all_images[docs_retrieved[0]["doc_id"]]

    # Create input messages
    system_prompt = "You are an AI assistant. Your task is reply to user questions based on the provided image context."
    chat_template = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": query}
            ]
        },
    ]

    # Prepare inputs
    prompt = processor_vlm.apply_chat_template(chat_template, add_generation_prompt=True)
    inputs = processor_vlm(text=prompt, images=[image_similar_to_query], return_tensors="pt")
    inputs = inputs.to(DEVICE)

    # Generate outputs
    generated_ids = model_vlm.generate(**inputs, max_new_tokens=500)
    #generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]

    generated_texts = processor_vlm.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    response = generated_texts[0]
    
    st.write(response)
