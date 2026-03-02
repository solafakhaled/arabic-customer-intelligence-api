from helpers.helper_cleaners import get_cleaned_data
from sklearn.model_selection import train_test_split

text= get_cleaned_data()
print(text[0:5])
# split data into training, testing and validation 

# train the model, vectorization and splitting at the same step 
