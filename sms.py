import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample SMS data
texts = [
    "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/12345 to claim now.",
    "Hey, are we still meeting for lunch tomorrow?",
    "Important notification: Your account has been suspended. Click the link to verify your details.",
    "Can you send me the report by today?",
    "Get a loan now! No credit check required. Apply at http://loanexample.com",
    "Happy Birthday! Have a wonderful day with lots of joy and fun!",
]

labels = ['spam', 'ham', 'spam', 'ham', 'spam', 'ham']

# Vectorize the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train the model
model = MultinomialNB()
model.fit(X, labels)

# Streamlit UI
st.title("SMS Spam Detection App")
st.write("Enter an SMS message to check if it's Ham or Spam:")

# Input text
user_input = st.text_area("Enter SMS Text Here", "")

if st.button("Predict"):
    if user_input:
        # Vectorize the input text
        input_data = vectorizer.transform([user_input])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Show the result
        st.subheader("Prediction:")
        st.success(f"This SMS is: **{prediction[0].upper()}**")
    else:
        st.warning("Please enter some text to predict.")






