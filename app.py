import uvicorn
import joblib
from fastapi import FastAPI ,HTTPException
from pydantic import BaseModel
import joblib
import re
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AdamWeightDecay, TFAutoModelForSeq2SeqLM

pipeline = joblib.load('language_detection_pipeline.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load the pre-trained model and tokenizer

model_eng2ar = TFAutoModelForSeq2SeqLM.from_pretrained("D:\Kemet NLP\en2ar_model")
tokenizer_eng2ar = AutoTokenizer.from_pretrained("D:\Kemet NLP\en2ar_tok")
 
model_ar2eng = TFAutoModelForSeq2SeqLM.from_pretrained("D:\Kemet NLP\ar2eng_model")
tokenizer_ar2eng =AutoTokenizer.from_pretrained("D:\Kemet NLP\ar2eng_model_tok")




def predict_pipeline(text):
    # Remove all the special characters
    text = re.sub(r'\W', ' ', str(text))
    # removing the symbols and numbers
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
    text = re.sub(r'\[\]', ' ', text)
    # Substituting multiple spaces with single space
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    # Removing prefixed 'b'
    text = re.sub(r'^b\s+', '', text)
    # Converting to Lowercase
    text = text.lower()
    
    predicted_language_encoded = pipeline.predict([text])[0]
    
    return predicted_language_encoded

def translate_eng2ar(clean_text):
    input_text =  clean_text
    inputs = tokenizer_eng2ar(input_text, return_tensors="pt").input_ids
    outputs = model_eng2ar.generate(inputs, max_length=64)
    translated_text = tokenizer_eng2ar.decode(outputs[0], skip_special_tokens=True)

    return translated_text

def translate_ar2eng(clean_text):
    input_text =  clean_text
    inputs = tokenizer_ar2eng(input_text, return_tensors="pt").input_ids
    outputs = model_ar2eng.generate(inputs, max_length=64)
    translated_text = tokenizer_ar2eng.decode(outputs[0], skip_special_tokens=True)

    return translated_text





class TextIn(BaseModel):
    TextIn: str

class PredictionOut(BaseModel):
    language: str
        
class Translation(BaseModel):
    language: str        
        

        
        
app = FastAPI()




@app.get("/")
def home():
    return {"health_check": "OK"}


@app.post("/predict", response_model=PredictionOut)
def predict_language(payload: TextIn):
    try:
        if not payload.TextIn.strip():
            raise HTTPException(status_code=400, detail="Empty text provided")
        
        predicted_language_encoded = predict_pipeline(payload.TextIn)
        predicted_language = label_encoder.inverse_transform([predicted_language_encoded])[0]
        predicted_language_str = str(predicted_language)  # Convert to string if necessary
        
        return {"language": predicted_language_str}
    
    except Exception as e:
        # Log the error
        print(f"An error occurred: {str(e)}")
        # Return an error response
        raise HTTPException(status_code=500, detail="Internal Server Error")

        
@app.post("/translation/" , response_model=Translation)
async def translate_text(text_data: TextIn):
    try:
        clean_text = text_data.TextIn.strip()
        if not clean_text:
            raise HTTPException(status_code=400, detail="Empty text provided")
        
        predicted_language_encoded = predict_pipeline(clean_text)
        predicted_language = label_encoder.inverse_transform([predicted_language_encoded])[0]
        predicted_language_str = str(predicted_language)
        
        if predicted_language_str == 'English':
            translated_text = translate_eng2ar(clean_text)
        elif predicted_language_str == 'Arabic':
            translated_text = translate_ar2eng(clean_text)
        else:
            raise HTTPException(status_code=400, detail="Unsupported language. Only Arabic and English are supported.")
        
        return {"translation": translated_text}  # Return only the translated text
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
