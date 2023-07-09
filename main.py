from sentence_transformers import SentenceTransformer
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
import pickle
df = pd.read_csv(r'C:/Users/athar/Downloads/Precily_Task/Precily_Task/Precily_Text_Similarity.csv')
def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.lower()
    return text

df['text1'] = df['text1'].apply(clean_text)
df['text2'] = df['text2'].apply(clean_text)
print(df.head())
model = SentenceTransformer('bert-base-nli-mean-tokens')  # Choose a suitable pre-trained model
text1_embeddings = model.encode(df['text1'], convert_to_tensor=True)
text2_embeddings = model.encode(df['text2'], convert_to_tensor=True)
similarity_scores = cosine_similarity(text1_embeddings, text2_embeddings)
normalized_scores = (similarity_scores - similarity_scores.min()) / (similarity_scores.max() - similarity_scores.min())
print(normalized_scores)
with open('sentence_transformer_model.pkl', 'wb') as f:
    pickle.dump(model, f)
