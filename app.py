from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
import json
from sentence_transformers import SentenceTransformer
# Load the saved model


app = Flask(__name__)

# Load the pre-trained model
model = SentenceTransformer('bert-base-nli-mean-tokens')
# with open('sentence_transformer_model.pkl', 'rb') as f:
#     model = pickle.load(f)


@app.route('/api/similarity', methods=['POST'])
def get_similarity_score():
    # Get the request body
    request_data = request.get_json()

    # Extract text1 and text2 from the request body
    text1 = request_data['text1']
    text2 = request_data['text2']
    # Encode text1 and text2 using the pre-trained model
    text1_embedding = model.encode(text1, convert_to_tensor=True)
    text2_embedding = model.encode(text2, convert_to_tensor=True)
    text1_embedding = np.reshape(text1_embedding, (1, -1))
    text2_embedding = np.reshape(text2_embedding, (1, -1))
    # Calculate the similarity score using cosine similarity
    similarity_score = cosine_similarity(text1_embedding, text2_embedding)[0][0]
    # out = "{:.1f}".format(similarity_score[0][1])
    # Prepare the response body
    a = round(similarity_score, 1)
    float_value = np.float32(a)
    float_value = float(float_value)
    response_body = {
        "similarity score": round(float_value, 1)
    }
    return jsonify(response_body)
if __name__ == '__main__':
    app.run()
