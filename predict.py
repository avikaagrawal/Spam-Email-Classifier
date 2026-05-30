import joblib

model = joblib.load("model/spam_classifier.pkl")

message = input("Enter a message: ")

prediction = model.predict([message])[0]

if prediction == 1:
    print("🚨 SPAM")
else:
    print("✅ NOT SPAM")