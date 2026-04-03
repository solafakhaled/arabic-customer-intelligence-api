#test the model like it is a user
import joblib

model = joblib.load('sentiment_model.joblib')

texts = [
    "الخدمة كانت ممتازة جدا وتجربة رائعة بصراحة",
    "تطبيق سيء جدا ومليان أخطاء، لا أنصح به أبدا",
    "عادي جدا، لا هو ممتاز ولا هو سيء"
]
predictions = model.predict(texts)

for text, pred in zip(texts, predictions):
    print(f"text: {text} | prediction: {pred}")

