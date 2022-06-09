from django import forms

class TwitterForm(forms.form):
    tweetid = forms.CharField(max_length=30)