import joblib
import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse
from .form import UserSignupForm
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from django.shortcuts import render

# Load the model, MultiLabelBinarizer, and preprocessor
model = joblib.load('tag_prediction_model.pkl')
mlb = joblib.load('mlb.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# Load your ad data (assuming it's in a CSV or similar format)
ad_data = pd.read_csv('EnrichedAdData.csv')  # Update with your actual path
vectorizer = TfidfVectorizer(lowercase=False)
ad_data['AdTagsString'] = ad_data['AdTags'].apply(lambda x: ''.join(x))
print("ad_data['AdTagsString']:",ad_data['AdTagsString'])
tfidf_matrix = vectorizer.fit_transform(ad_data['AdTagsString'])

def signup_view(request):
    if request.method == 'POST':
        form = UserSignupForm(request.POST)
        if form.is_valid():
            user_data = form.cleaned_data
            
            # Function to calculate age group based on DOB
            def calculate_age_group(dob):
                current_year = datetime.now().year
                dob_year = pd.to_datetime(dob).year
                age = current_year - dob_year
                if 18 <= age <= 24:
                    return "18-24"
                elif 25 <= age <= 30:
                    return "25-30"
                elif 31 <= age <= 45:
                    return "31-45"
                else:
                    return "46-above"

            # Calculate the age group using the DOB from user_data
            age_group = calculate_age_group(user_data['dob'])
            print("age_group", age_group)

            # Prepare the new user data for prediction
            new_user_data = pd.DataFrame({
                'Age Group': [age_group],  # Include Age Group
                'Gender': [user_data['gender']],
                'Payment Information': [user_data['payment_info']],
                'Usage Frequency': [user_data['usage_frequency']],
                'Purchase History': [user_data['purchase_history']],
                'Favorite Genres': [user_data['favorite_genre']],
                'Devices Used': [user_data['device_used']],
                'Engagement Metrics': [user_data['engagement_metrics']]
            })

            print("new_user_data:", new_user_data)

            # Call the predict_tags function
            predicted_tags = predict_tags(new_user_data)
            print("Predicted Tags:", predicted_tags)

            # Generate ads based on predicted tags
            user_interests = ' '.join(predicted_tags[0])  # Join the predicted tags into a single string
            user_vector = vectorizer.transform([user_interests])

            # Compute cosine similarity between user interests and ad tags
            cosine_similarities = cosine_similarity(user_vector, tfidf_matrix)

            # Get the indices of the top 5 ads
            top_indices = cosine_similarities[0].argsort()[-5:][::-1]

            # Retrieve the top ads
            top_ads = ad_data.iloc[top_indices][['AdID', 'AdTitle', 'AdLink']]

            # Prepare the response
            # Prepare the response data
            response_data = {
                'status': 'success',
                'data': user_data,
                'predicted_tags': predicted_tags,
                'top_ads': top_ads.to_dict(orient='records')  # Convert DataFrame to a list of dictionaries
            }

            # Render the results in an HTML template
            return render(request, 'results.html', {'response': response_data})

        else:
            return JsonResponse({'status': 'error', 'errors': form.errors}, status=400)
    else:
        form = UserSignupForm()
    return render(request, 'signup.html', {'form': form})

def predict_tags(new_user_data):
    print("Inside predict tags!")
    
    # Define categorical columns
    categorical_columns = ['Age Group', 'Gender', 'Payment Information', 'Usage Frequency', 'Purchase History', 'Favorite Genres', 'Devices Used', 'Engagement Metrics']
    
    # Ensure the new user data is in the correct format
    try:
        # Ensure that new_user_data contains all required columns
        print("new_user_data columns:", new_user_data.columns)
        print("new_user_data[categorical_columns]:", new_user_data[categorical_columns])
        
        # Transform categorical data
        encoded_new_data = preprocessor.transform(new_user_data[categorical_columns])
        
        # Since there are no numerical columns, we can skip creating new_numerical_data
        new_features = encoded_new_data  # Use only the encoded categorical data
        print("new_features shape:", new_features.shape)
    except Exception as e:
        print("Error during preprocessing:", e)
        return []

    # Make predictions
    predictions = model.predict(new_features)

    # Decode the predicted tags
    predicted_tags = mlb.inverse_transform(predictions)
    return predicted_tags 


def results(request):
    # Mock response (replace this with your actual logic)
    response_data = {
        "status": "success",
        "data": {
            "dob": "1998-02-10",
            "gender": "Male",
            "location": "Pune",
            "payment_info": "Mastercard",
            "usage_frequency": "Frequent",
            "purchase_history": "Electronics",
            "favorite_genre": "Horror",
            "device_used": "SmartPhone",
            "engagement_metrics": "High"
        },
        "predicted_tags": [
            [
                "Male", "active user", "adventurous", "dedicated", 
                "innovation lover", "professional", "settling down", 
                "tech enthusiast", "tech-savvy", "thrill-seeker"
            ]
        ],
        "top_ads": [
            {
                "AdID": 1,
                "AdTitle": "Amazon Shopping App",
                "AdLink": "https://imgs.search.brave.com/T5LGtSBmdglkozcbQBjBwN2YquDpSFYVew0f6DMdwUs/rs:fit:500:0:0:0/g:ce/aHR0cHM6Ly92aXN1/YWxoaWVyYXJjaHku/Y28vd3AtY29udGVu/dC91cGxvYWRzLzIw/MTgvMDQvYW1hem9u/LmpwZw"
            },
            {
                "AdID": 4,
                "AdTitle": "Spotify Music App",
                "AdLink": "https://imgs.search.brave.com/BfVJQVoG-i_FoTTXM8oWoNGa9ioREWM6rK4Jjh_GzOY/rs:fit:500:0:0:0/g:ce/aHR0cHM6Ly9icmFu/ZGluZ2ZvcnVtLm9y/Zy93cC1jb250ZW50/L3VwbG9hZHMvMjAy/My8xMC9zcG90aWZ5/LWxvZ28tMTAyNHg2/NTEud2VicA"
            },
            {
                "AdID": 26,
                "AdTitle": "Hotstar Disney+",
                "AdLink": "https://imgs.search.brave.com/mHYRI3f5OHMxjv9hqyeDSt6SjMEkGGvaBcdKQDzBsKE/rs:fit:500:0:0:0/g:ce/aHR0cHM6Ly9ha20t/aW1nLWEtaW4udG9z/c2h1Yi5jb20vYnVz/aW5lc3N0b2RheS9p/bWFnZXMvc3Rvcnkv/MjAyMDAzL2Rpc25l/eWhvdHN0YXJfNjYw/XzExMDMyMDAzNDQy/OC5qcGc_c2l6ZT05/NDg6NTMz"
            },
            {
                "AdID": 11,
                "AdTitle": "Samsung Galaxy Phones",
                "AdLink": "https://imgs.search.brave.com/VvIzrbScKRcG_YIUL47R19NjUKndFunPBLzyBJjX8Qc/rs:fit:500:0:0:0/g:ce/aHR0cHM6Ly9sb2dv/d2lrLmNvbS9jb250/ZW50L3VwbG9hZHMv/aW1hZ2VzL3NhbXN1/bmczNTMwOS5sb2dv/d2lrLmNvbS53ZWJw"
            },
            {
                "AdID": 14,
                "AdTitle": "Oppo Smartphones",
                "AdLink": "https://imgs.search.brave.com/KlQAmTBS4ppAvbxdMJrPT_zOaZxCo2EnNceGJBeeCO0/rs:fit:500:0:0:0/g:ce/aHR0cHM6Ly9sb2dv/d2lrLmNvbS9jb250/ZW50L3VwbG9hZHMv/aW1hZ2VzL29wcG8u/anBn"
            }
        ]
    }
    return render(request, 'results.html', {'response': response_data})