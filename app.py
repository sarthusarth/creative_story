import streamlit as st
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.experimental_singleton
def load_model():
    path = 'isarth/distill_gpt2_story_generator'
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path)  
    return  model, tokenizer

@st.experimental_singleton
def image_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    
    model.to(device)
    return  model, tokenizer, feature_extractor

@st.cache
def generate_caption(image_paths, model, tokenizer, feature_extractor):
    print('Generating captions')
    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds


def generate_story(prompt, tokenizer, model):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, do_sample=True, max_length=150, top_p=0.95)
    ans = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return ans[:-1*ans[::-1].find('.')]

model, tokenizer, feature_extractor = image_model()
lm_model, lm_tokenizer = load_model()

st.title("Creative Story Generation")
history = {}
uploaded_file = st.file_uploader("Upload the Image")
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, width=500)
    if str(uploaded_file) in list(history.keys()):
        cap = history[str(uploaded_file)]
    else:
        cap = generate_caption([uploaded_file],  model, tokenizer, feature_extractor)[0]
        history[str(uploaded_file)] = cap    
    st.write(cap)
    option = st.selectbox('Select a Genre?',('drama', 'horror', 'superhero','action','thriller','sci_fi'))
    if st.button('run'):
        with st.spinner(f'Generating a {option} story ...'):
            prompt = f'<BOS> <{option}> {cap}'
            story = generate_story(prompt, lm_tokenizer, lm_model)
        st.success('Done!')
        st.write(story)

