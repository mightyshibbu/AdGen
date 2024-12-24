import joblib
import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse
from .form import UserSignupForm
import numpy as np
from datetime import datetime
# Load the model, MultiLabelBinarizer, and preprocessor
model = joblib.load('tag_prediction_model.pkl')
mlb = joblib.load('mlb.pkl')
preprocessor = joblib.load('preprocessor.pkl')
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
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PREDICTED TAGS ARE HERE!
            
            return JsonResponse({'status': 'success', 'data': user_data, 'predicted_tags': predicted_tags})
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