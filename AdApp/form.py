# myapp/forms.py
from django import forms

class UserSignupForm(forms.Form):
    dob = forms.DateField(label='Date of Birth', widget=forms.DateInput(attrs={'type': 'date'}))
    gender = forms.ChoiceField(choices=[('Male', 'Male'), ('Female', 'Female'), ('Others', 'Others')])
    location = forms.CharField(max_length=100)
    payment_info = forms.ChoiceField(choices=[('Mastercard', 'Mastercard'), ('VISA', 'VISA'), ('Amex', 'Amex')])
    usage_frequency = forms.ChoiceField(choices=[('Frequent', 'Frequent'), ('Regular', 'Regular'), ('Occasional', 'Occasional')])
    purchase_history = forms.ChoiceField(choices=[('Electronics', 'Electronics'), ('Books', 'Books'), ('Clothings', 'Clothings')])
    favorite_genre = forms.ChoiceField(choices=[('Horror', 'Horror'), ('Sci-fi', 'Sci-fi'), ('Comedy', 'Comedy'), ('Documentary', 'Documentary'), ('Drama', 'Drama'), ('Action', 'Action'), ('Romance', 'Romance')])
    device_used = forms.ChoiceField(choices=[('SmartPhone', 'SmartPhone'), ('SmartTV', 'SmartTV'), ('Tablet', 'Tablet')])
    engagement_metrics = forms.ChoiceField(choices=[('High', 'High'), ('Medium', 'Medium'), ('Low', 'Low')])