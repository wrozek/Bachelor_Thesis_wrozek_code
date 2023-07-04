import nltk
import spacy as spacy
from nltk.corpus import stopwords
import lxml.html
import string
import pandas as pd
from nltk import word_tokenize

# Read the data from the swissdox database
data = pd.read_csv('Updated_dataset__2023_02_09.tsv',sep='\t', header = 0)

# Drop identical rows by looking at the header and the content & reset index
def drop_identical_rows(df, subset_columns):
    df = df.drop_duplicates(subset=subset_columns, keep="first")
    df = df.reset_index(drop=True)
    return df
data = drop_identical_rows(data, "content")
data = drop_identical_rows(data, "head")


# Remove html tags from content
def extract_html_tags(dataframe, column_name):
    datacon = dataframe[column_name].values
    data_1 = []
    for i in datacon:
        datacon = lxml.html.fromstring(i).text_content()
        data_1.append(datacon)
    data_content = pd.DataFrame(data_1, columns=[column_name])
    dataframe.loc[:, column_name] = data_content.loc[:, column_name]
    #dataframe[column_name] = data_content[column_name]
    return dataframe

data = extract_html_tags(data, "content")

#lowercasing the headers and content
def lowercase_columns(df, columns):
    for column in columns:
        df[column] = df[column].str.lower()
    return df

data = lowercase_columns(data, ["head", "content"])

# Remove again identical rows by looking at the content column
data = drop_identical_rows(data, "content")
data = drop_identical_rows(data, "head")

#remove punctuation in "content column"
def remove_punctuation(text):
    translator = str.maketrans("", "", string.punctuation + '«'+ '»'+ '—'+ '–'+ '‘'+ '’'+ '“'+ '”')
    return text.translate(translator)

data["content"] = data["content"].apply(remove_punctuation)


#define stopwords
stop_words = set()
for language in ['english', 'german']:
    stop_words.update(stopwords.words(language))
#add more stopwords
additional_stopwords = ["wurde", "sei", "seit", "bereits", "sagte", "mehr", "zwei", "rund", "ersten", "beim", "immer", "schon", "sagt", "gibt"]
stop_words.update(additional_stopwords)

#copy the dataset
data_stopwords = data.copy()

#remove stopwords from dataframe column - important for next steps like tokenization or lemmatization
def remove_stop_words(data, columns):
    for column in columns:
        data[column] = data[column].str.split(' ').apply(lambda x: [word for word in x if word not in stop_words])
        data[column] = data[column].str.join(' ')
    return data

data_stopwords = remove_stop_words(data_stopwords, ["head", "content"])

#Make a dataset only with refugee content
def find_pattern_in_column(df, column_name, pattern):
    return df[df[column_name].str.contains(pattern, flags=nltk.re.IGNORECASE, regex=True)]

pattern = "flüchtling|refugee|fugitive|flüchtlinge"
data_refugee = find_pattern_in_column(data, "content", pattern)

#create a dataset of only 1 column called "Test" containing the "content" column
column_name = data_refugee['content']  # select the column and slice to contain only first 100 rows
new_df = pd.DataFrame({'Test': column_name})  # create a new dataframe with the selected column

#tokenize df column considering english and german
def tokenize_df_column(df, text_column, language_column):
    df_tokenized = df.copy()
    df_tokenized['tokens'] = df_tokenized.apply(lambda row: word_tokenize(row[text_column]) if row[language_column] == 'en' else word_tokenize(row[text_column], language='german'), axis=1)
    return df_tokenized

data_refugee = tokenize_df_column(data_refugee, 'content', 'language')


#lemmatization of df column considering english and german with the spacy modul as the nltk module does not support german
def lemmatize_column(df, text_column, language_column):
    # Load the appropriate spaCy model based on the language column
    models = {'en': 'en_core_web_sm', 'de': 'de_core_news_sm'}
    model_name = models[df[language_column].iloc[0]]
    nlp = spacy.load(model_name)

    # Create a list to hold the lemmatized text for each row
    lemmatized_text = []

    # Use nlp.pipe() to process the text in batches
    for doc in nlp.pipe(df[text_column].values):
        lemmatized_tokens = [token.lemma_ for token in doc]
        lemmatized_text.append(lemmatized_tokens)

    # Add the lemmatized text to the dataframe
    df["lemmas"] = lemmatized_text

    return df
data_refugee = lemmatize_column(data_refugee, "content", "language")

#Drop unnecessary columns
def drop_columns(df, columns_to_drop):
    df = df.drop(columns_to_drop, axis=1)
    return df
data_refugee = drop_columns(data_refugee, ['id', 'medium_code', 'medium_name', 'rubric', 'regional', 'doctype', 'doctype_description', 'dateline', 'subhead', 'content_id'])

#reset index
data_refugee=data_refugee.reset_index(drop=True)

# Write the DataFrame to a TSV file
data_refugee.to_csv('data_refugee_SA.tsv', sep='\t', index=False)


### Sort the dataframe for SA according to the publication time ###
data = pd.read_csv('data_refugee_SA.tsv',sep='\t', header = 0)
data['pubtime'] = pd.to_datetime(data['pubtime'])
data = data.sort_values(by='pubtime', ascending=True)
data=data.reset_index(drop=True)
data.to_csv('data_refugee_SA_sorted.tsv', sep='\t', index=False)