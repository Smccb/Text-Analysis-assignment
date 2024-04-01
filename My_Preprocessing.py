import re
import pandas as pd
import json
import emoji
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from imblearn.over_sampling import SMOTE, RandomOverSampler

#####################################################################

#Remove #sarcasm
def remove_Sarcasm_hashtag():
    file_path = 'Train_v1.txt'
    column_names = ['toRemove', 'isSarcastic', 'text']

    # Read the dataset
    data = pd.read_csv(file_path, sep='\t', header=None, names=column_names)

    # Define patterns to remove
    patterns = [
        r'#sarcasm\b',
        r'#sarcastic\b',
        r'#sarcastictweet\b'
    ]

    # Remove patterns from text
    for pattern in patterns:
        data['text'] = data['text'].apply(lambda x: re.sub(pattern, '', x))

    # Drop the 'toRemove' column
    data.drop(columns=['toRemove'], inplace=True)

    # Convert DataFrame to dictionary
    data_dict = data.to_dict()

    # Save the dictionary to a JSON file
    with open('cleaned_#sarcasm.json', 'w') as f:
        json.dump(data_dict, f, indent=4)

#remove_Sarcasm_hashtag()

######################################################################

#update number of rows in dataset undersampling
def undersampling(dataset):
    # Count the number of instances in each class
    class_counts = dataset['isSarcastic'].value_counts()

    # Find the class with more items
    majority_class = class_counts.idxmax()

    # Find the class with fewer items
    minority_class = class_counts.idxmin()

    # Count the number of instances in the minority class
    minority_class_count = class_counts[minority_class]

    majority_class_sampled = dataset[dataset['isSarcastic'] == majority_class].sample(n=minority_class_count, random_state=42)

    # Concatenate the sampled majority class with the minority class
    balanced_data = pd.concat([majority_class_sampled, dataset[dataset['isSarcastic'] == minority_class]])

    # Shuffle the balanced dataset
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_data

######################################################################

#Oversampling

#Random
def random_oversampling(dataset):
    # Initialize RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    # Resample the dataset
    X_resampled, y_resampled = ros.fit_resample(dataset.drop(columns=['isSarcastic']), dataset['isSarcastic'])

    # Combine resampled features and target variable into a DataFrame
    resampled_df = pd.DataFrame(X_resampled, columns=dataset.drop(columns=['isSarcastic']).columns)
    resampled_df['isSarcastic'] = y_resampled

    return resampled_df

#SMOTE sampling
def smote_oversampling(dataset):
    # Initialize SMOTE
    smote = SMOTE(random_state=42)
    # Resample the dataset
    X_resampled, y_resampled = smote.fit_resample(dataset.drop(columns=['isSarcastic']), dataset['isSarcastic'])

    # Combine resampled features and target variable into a DataFrame
    resampled_df = pd.DataFrame(X_resampled, columns=dataset.drop(columns=['isSarcastic']).columns)
    resampled_df['isSarcastic'] = y_resampled

    return resampled_df

#####################################################################


# Functions to remove hashtags from a single text
def remove_hashtags(text):
    pattern = r'\#\w+'
    return re.sub(pattern, '', text)

#call this one
def hashtags(dataset):
    
    dataset['text'] = dataset['text'].apply(remove_hashtags)
    data_dict = dataset.to_dict()
    with open('all#Removed.json', 'w') as f:
        json.dump(data_dict, f, indent=4)

    return dataset

################################################################

def replace_abbreviations(dataset):
    abbreviation_mapping = {
        'OMG': 'oh my god',
        'DM': 'direct message',
        'BTW': 'by the way',
        'BRB': 'be right back',
        'RT': 'retweet',
        'FTW': 'for the win',
        'QOTD': 'quote of the day',
        'IDK': 'I do not know',
        'ICYMI': 'in case you missed it',
        'IRL': 'in real life',
        'IMHO': 'in my humble opinion',
        'IMO': 'I do not know',
        'LOL': 'laugh out loud',
        'LMAO': 'laughing my ass off',
        'LMFAO': 'laughing my fucking ass off',
        'NTS': 'note to self',
        'F2F': 'face to face',
        'B4': 'before',
        'DM': 'direct message',
        'CC': 'carbon copy',
        'SMH': 'shaking my head',
        'STFU': 'shut the fuck up',
        'BFN': 'by for now',
        'AFAIK': 'as far as I know',
        'TY': 'thank you',
        'YW': 'you are welcome',
        'THX': 'thanks'
    }

    # Function to replace abbreviations
    def replace_abbreviations(text):
        tokens = text.split()
        for i, token in enumerate(tokens):
            if token.upper() in abbreviation_mapping:
                tokens[i] = abbreviation_mapping[token.upper()]
        return ' '.join(tokens)

    # Apply functions to remove hashtags and replace abbreviations to the entire 'text' column
    dataset['text'] = dataset['text'].apply(lambda x: x.upper())  # Convert text to uppercase
    dataset['text'] = dataset['text'].apply(replace_abbreviations)

    # Restore original capitalization
    original_capitalization = lambda x: ''.join([a if b.islower() else a.lower() for a, b in zip(x, dataset['text'][0])])
    dataset['text'] = dataset['text'].apply(original_capitalization)

    # Save the updated DataFrame to a JSON file
    dataset.to_json('abbreviations_removed.json', orient='records', lines=True)
    return dataset

##################################################################

#replace @ with person
def replace_user_mentions(dataset):

    def remove_user_mentions(text):
        pattern = re.compile(r'@\d+')
        return pattern.sub('person', text)

    # Apply remove_user_mentions function to the 'text' column
    dataset['text'] = dataset['text'].apply(remove_user_mentions)
    # Save the updated DataFrame if needed
    dataset.to_json("updated_data_without_mentions.json", orient='records', lines=True)

    return dataset

###################################################################

def replace_emojis(text):
    return emoji.demojize(text, delimiters=(" ", " "))


def replace_emoticons(text):
    emoticon_dict = {
    ':)': 'smile',
    ':(': 'frown',
    ':D': 'big smile',
    ':P': 'tongue out',
    ';)': 'wink',
    ':O': 'surprise',
    ':|': 'neutral',
    ':/': 'uncertain',
    ":'(": 'tears of sadness',
    ":'D": 'tears of joy',
    ':*': 'kiss',
    ':@': 'angry',
    ':x': 'mouth shut',
    ':3': 'cute',
    ':$': 'embarrassed',
    ":')": 'single tear',
    ':p': 'tongue out'
}
    emoticon_dict_lower = {key.lower(): value for key, value in emoticon_dict.items()}

    pattern = re.compile(r'(' + '|'.join(re.escape(emoticon) for emoticon in emoticon_dict_lower.keys()) + ')', re.IGNORECASE)

    return pattern.sub(lambda match: emoticon_dict_lower.get(match.group().lower(), match.group()), text)

#replace emoji and emoticon
def replace_emoji_emoticons(dataset):
    dataset['text'] = dataset['text'].apply(replace_emojis)
    dataset['text'] = dataset['text'].apply(replace_emoticons)

    data_dict = dataset.to_dict()

    with open('removedEmoji.json', 'w') as f:
        json.dump(data_dict, f, indent=4)

    return dataset

######################################################################

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

#stopwords removed
def stop_words(dataset):
    # Apply remove_stopwords function to the 'text' column
    dataset['text_without_stopwords'] = dataset['text'].apply(remove_stopwords)
    return dataset

