from datasets import load_dataset
import nltk
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
import re

def normalize_arabic(text):
    # 1. Normalize Alef variants to simple Alef
    text = re.sub(r"[أإآ]", "ا", text)
    # 2. Normalize Taa Marbuta to Ha
    text = re.sub(r"ة", "ه", text)
    # 3. Normalize Alef Maqsura to Ya
    text = re.sub(r"ى", "ي", text)
    # 4. (Optional) Remove Tatweel (elongation like ـ)
    text = re.sub(r"ـ", "", text)
    # 5. Remove diacritics (Tashkeel)
    re.sub(r'[\u064B-\u065F\u0670]', '', text)
    # 6. Remove Emogies and non arabic letters. 
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[^\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff\ufe70-\ufeff\s]', '', text)
    # 7. Remove repeated letters (e.g., "جميلللل" to "جميل") more than twice 
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    # 8. Remove extra spaces or newlines or tabs 
    text = re.sub(r'\s+', ' ', text).strip()

    
    return text


# call create data 
train = load_dataset("josephnashat/ArabicSentimentAnalysis", split="train")
#print("This is the train split ",train)

# normalize data 
data = train.map(lambda x: {"text": normalize_arabic(x["text"])})
#print(len(data))
#print(data[0:5])

# Tokenization + Stop Words Removal + Stemming.
#get the set of stopwords 
nltk.download('stopwords')
stemmer= ISRIStemmer()
arabic_stopwords = set([normalize_arabic(word) for word in stopwords.words('arabic')])

def preprocessing_tokens(text):
    tokens = text.split()
    # remove stop words
    tokens = [word for word in tokens if word not in arabic_stopwords]
    # stemming
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

data = data.map(lambda x: {"text": preprocessing_tokens(x["text"])})
#print(data[0:5])
    
def get_cleaned_data():
    return data


# ok, this is what the data looks like 
#Dataset({
#    features: ['text', 'polarity'],
#    num_rows: 964 })
# We need to perform the split using the helper function, because this dataset includes only training split. aka it is one big chunk
# we will split it into 60% train and 20% test and 20% validation.

#lets print the first few rows of the dataset to understand it better

# the data includes 964 row paragraphs in arabic and a polarity label of 0,1, -1
#If you use Transformers (like AraBERT/mBERT): You generally SKIP stemming and stop-word removal. Why? Because Transformers need the "context" (the "to", "in", "the") to understand the sentence structure. If you stem words to their roots, you lose the grammar that BERT relies on.
#If you use Classic ML (like SVM, Naive Bayes, Random Forest): Then yes, Stemming and Lemmatization are crucial because these models treat words as dumb "counts" (Bag of Words) and need to know that "playing" and "play" are the same thing.
# I am learning step by step, so I will do the same project with three versions
# 1. Classic ML with Stemming and Stop-word removal
# 2. Transformers without Stemming and Stop-word removal
# 3. Using LLMs with prompt engineering (no training, just inference) just to learn.
# lets start with the classical version which requires the use of bag of words, stemming, lemmatization and stop word removal, POS and NEM.
