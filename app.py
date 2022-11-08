import streamlit as st 
import requests
from PIL import Image
from transformers import ViTFeatureExtractor, AutoTokenizer, FlaxVisionEncoderDecoderModel


loc = "ydshieh/vit-gpt2-coco-en"

feature_extractor = ViTFeatureExtractor.from_pretrained(loc)
tokenizer = AutoTokenizer.from_pretrained(loc)
model = FlaxVisionEncoderDecoderModel.from_pretrained(loc)












def main_loop():
    st.title("OpenCV Demo App")
    st.subheader("This app allows you to play with Image filters!")
    st.text("We use OpenCV and Streamlit for this demo")
    button = st.button("Analyze")
    
    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None

    original_image = Image.open(image_file)
    pixel_values = feature_extractor(images=original_image, return_tensors="np").pixel_values
    button = st.button("Analyze")
    def generate_step(pixel_values):

        output_ids = model.generate(pixel_values, max_length=16, num_beams=4).sequences
        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        
        return preds
    

    preds = generate_step(pixel_values)
    st.write('Output', preds)
    
if __name__ == '__main__':
    main_loop()    
    
    
    
 