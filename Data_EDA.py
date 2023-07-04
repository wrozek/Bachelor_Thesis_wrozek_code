import numpy as np
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

#Read the datasets with lemmatized and tokenized columns
data_original_refugee = pd.read_csv('data_stopwords_refugee1.tsv',sep='\t', header = 0)
#data_original_refugee = pd.read_csv('data_refugee1.tsv',sep='\t', header = 0)

#perform bag of words and print the most frequent words
exclude_words = ["russischen", "russische", "ukrainischen", "ukrainische", "seien", "worden", "wurden", "angaben", "viele", "wir", "russisch", "ukrainisch", "werden", "xa0", "sein", "neu", "kommen", "weit", "gehen", "sollen", "vieler", "Angabe", "geben", "können", "stehen", "gut", "haben", "müssen", "Person", "Prozent", "weitere", "etwa", "laut", "wegen", "die" ]
def bag_of_words(df, text_column, n=10):
    vectorizer = CountVectorizer(lowercase=False, stop_words=exclude_words)
    bow = vectorizer.fit_transform(df[text_column].tolist())
    features = vectorizer.get_feature_names_out()
    sum_words = bow.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in zip(features, range(len(features)))]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

#Plot bag of words
def plot_bag_of_words(words_freq, N):
    words = [word for word, freq in words_freq[:N]]
    freqs = [freq for word, freq in words_freq[:N]]

    plt.bar(words, freqs)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Most Common Words')
    plt.xticks(rotation=90)
    plt.show()

plot_bag_of_words(bag_of_words(data_original_refugee, "lemmas", 20), 20)

#create tf-idf matrix and get the N top words for each document
def create_tfidf_matrix(dataframe, column_name):
    # convert the text column to a list of strings
    text_column = dataframe[column_name].tolist()
    # initialize the vectorizer
    vectorizer = TfidfVectorizer(lowercase=False)
    # fit and transform the text column
    tfidf_matrix = vectorizer.fit_transform(text_column)
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names

tf_test, feature_names=create_tfidf_matrix(data_original_refugee, "head")

#get the top words for each document
def get_top_words(tfidf_matrix, feature_names, N):
    top_words_indices = np.zeros((tfidf_matrix.shape[0], N), dtype=np.int64)

    for i in range(tfidf_matrix.shape[0]):
        # get the indices of the top N words for this document
        top_words_indices[i, :] = np.argsort(tfidf_matrix[i, :].toarray())[0, -N:]

    # create a list to store the top words for each document
    top_words = []
    for row in top_words_indices:
        # get the words with the highest TF-IDF values for this document
        document_top_words = [feature_names[index] for index in row]
        top_words.append(document_top_words)

    return top_words

top_words = get_top_words(tf_test, feature_names, 3)

#visualize top_words
def visualize_top_words(top_words, top_n =3, num_docs_to_show=None):
    top_words_dict = {}
    for i, words in enumerate(top_words):
        top_words_dict[i] = words

    num_docs = len(top_words)
    if num_docs_to_show:
        num_docs = min(num_docs, num_docs_to_show)
        top_words = top_words[:num_docs]
        top_words_dict = {k: v for k, v in top_words_dict.items() if k < num_docs}
    bar_width = 0.35
    x = np.arange(num_docs)
    y = [len(words) for words in top_words]
    # get the number of documents

    fig, ax = plt.subplots()
    bars = ax.bar(x, y, bar_width, align='center')
    #plt.bar(x, y, bar_width, align='center')
    plt.xticks(x, top_words_dict.keys())
    plt.yticks(y, y)
    plt.title('Top Words in Each Document')
    # Add the top words as annotations
    for i, bar in enumerate(bars):
        ax.annotate(", ".join(top_words[i]),
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(-25, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='top', rotation="vertical", fontsize=20)

    plt.show()

visualize_top_words(top_words, num_docs_to_show=15)

# Build a wordcloud
def wordcloud(data):
    # add german stopwords
    stopwords_de = ["a", "ab", "aber", "ach", "acht", "achte", "achten", "achter", "achtes", "ag", "alle", "allein", "allem", "allen", "aller", "allerdings", "alles", "allgemeinen", "als", "also", "am", "an", "andere", "anderen", "andern", "anders", "au", "auch", "auf", "aus", "ausser", "außer", "ausserdem", "außerdem", "b", "bald", "bei", "beide", "beiden", "beim", "beispiel", "bekannt", "bereits",
                    "besonders", "besser", "besten", "bin", "bis", "bisher", "bist", "c", "d", "da", "dabei", "dadurch",
                    "dafür", "dagegen", "daher", "dahin", "dahinter", "damals", "damit", "danach", "daneben", "dank",
                    "dann", "daran", "darauf", "daraus", "darf", "darfst", "darin", "darüber", "darum", "darunter",
                    "das", "dasein", "daselbst", "dass", "daß", "dasselbe", "davon", "davor", "dazu", "dazwischen",
                    "dein", "deine", "deinem", "deiner", "dem", "dementsprechend", "demgegenüber", "demgemäss",
                    "demgemäß", "demselben", "demzufolge", "den", "denen", "denn", "denselben", "der", "deren",
                    "derjenige", "derjenigen", "dermassen", "dermaßen", "derselbe", "derselben", "des", "deshalb",
                    "desselben", "dessen", "deswegen", "d.h", "dich", "die", "diejenige", "diejenigen", "dies", "diese",
                    "dieselbe", "dieselben", "diesem", "diesen", "dieser", "dieses", "dir", "doch", "dort", "drei",
                    "drin", "dritte", "dritten", "dritter", "drittes", "du", "durch", "durchaus", "dürfen", "dürft",
                    "durfte", "durften", "e", "eben", "ebenso", "ehrlich", "ei", "ei,", "eigen", "eigene", "eigenen",
                    "eigener", "eigenes", "ein", "einander", "eine", "einem", "einen", "einer", "eines", "einige",
                    "einigen", "einiger", "einiges", "einmal", "eins", "elf", "en", "ende", "endlich", "entweder", "er",
                    "Ernst", "erst", "erste", "ersten", "erster", "erstes", "es", "etwa", "etwas", "euch", "f",
                    "früher", "fünf", "fünfte", "fünften", "fünfter", "fünftes", "für", "g", "gab", "ganz", "ganze",
                    "ganzen", "ganzer", "ganzes", "gar", "gedurft", "gegen", "gegenüber", "gehabt", "gehen", "geht",
                    "gekannt", "gekonnt", "gemacht", "gemocht", "gemusst", "genug", "gerade", "gern", "gesagt",
                    "geschweige", "gewesen", "gewollt", "geworden", "gibt", "ging", "gleich", "gott", "gross", "groß",
                    "grosse", "große", "grossen", "großen", "grosser", "großer", "grosses", "großes", "gut", "gute",
                    "guter", "gutes", "h", "habe", "haben", "habt", "hast", "hat", "hatte", "hätte", "hatten", "hätten",
                    "heisst", "her", "heute", "hier", "hin", "hinter", "hoch", "i", "ich", "ihm", "ihn", "ihnen", "ihr",
                    "ihre", "ihrem", "ihren", "ihrer", "ihres", "im", "immer", "in", "indem", "infolgedessen", "ins",
                    "irgend", "ist", "j", "ja", "jahr", "jahre", "jahren", "je", "jede", "jedem", "jeden", "jeder",
                    "jedermann", "jedermanns", "jedoch", "jemand", "jemandem", "jemanden", "jene", "jenem", "jenen",
                    "jener", "jenes", "jetzt", "k", "kam", "kann", "kannst", "kaum", "kein", "keine", "keinem",
                    "keinen", "keiner", "kleine", "kleinen", "kleiner", "kleines", "kommen", "kommt", "können", "könnt",
                    "konnte", "könnte", "konnten", "kurz", "l", "lang", "lange", "leicht", "leide", "lieber", "los",
                    "m", "machen", "macht", "machte", "mag", "magst", "mahn", "man", "manche", "manchem", "manchen",
                    "mancher", "manches", "mann", "mehr", "mein", "meine", "meinem", "meinen", "meiner", "meines",
                    "mensch", "menschen", "mich", "mir", "mit", "mittel", "mochte", "möchte", "mochten", "mögen",
                    "möglich", "mögt", "morgen", "muss", "muß", "müssen", "musst", "müsst", "musste", "mussten", "n",
                    "na", "nach", "nachdem", "nahm", "natürlich", "neben", "nein", "neue", "neuen", "neun", "neunte",
                    "neunten", "neunter", "neuntes", "nicht", "nichts", "nie", "niemand", "niemandem", "niemanden",
                    "noch", "nun", "nur", "o", "ob", "oben", "oder", "offen", "oft", "ohne", "Ordnung", "p", "q", "r",
                    "recht", "rechte", "rechten", "rechter", "rechtes", "richtig", "rund", "s", "sa", "sache", "sagt",
                    "sagte", "sah", "satt", "schlecht", "Schluss", "schon", "sechs", "sechste", "sechsten", "sechster",
                    "sechstes", "sehr", "sei", "seid", "seien", "sein", "seine", "seinem", "seinen", "seiner", "seines",
                    "seit", "seitdem", "selbst", "sich", "sie", "sieben", "siebente", "siebenten", "siebenter",
                    "siebentes", "sind", "so", "solang", "solche", "solchem", "solchen", "solcher", "solches", "soll",
                    "sollen", "sollte", "sollten", "sondern", "sonst", "sowie", "später", "statt", "t", "tag", "tage",
                    "tagen", "tat", "teil", "tel", "tritt", "trotzdem", "tun", "u", "über", "überhaupt", "übrigens",
                    "uhr", "um", "und", "und?", "uns", "unser", "unsere", "unserer", "unter", "v", "vergangenen",
                    "viel", "viele", "vielem", "vielen", "vielleicht", "vier", "vierte", "vierten", "vierter",
                    "viertes", "vom", "von", "vor", "w", "wahr?", "während", "währenddem", "währenddessen", "wann",
                    "war", "wäre", "waren", "wart", "warum", "was", "wegen", "weil", "weit", "weiter", "weitere",
                    "weiteren", "weiteres", "welche", "welchem", "welchen", "welcher", "welches", "wem", "wen", "wenig",
                    "wenige", "weniger", "weniges", "wenigstens", "wenn", "wer", "werde", "werden", "werdet", "wessen",
                    "wie", "wieder", "will", "willst", "wir", "wird", "wirklich", "wirst", "wo", "wohl", "wollen",
                    "wollt", "wollte", "wollten", "worden", "wurde", "würde", "wurden", "würden", "x", "y", "z", "z.b",
                    "zehn", "zehnte", "zehnten", "zehnter", "zehntes", "zeit", "zu", "zuerst", "zugleich", "zum",
                    "zunächst", "zur", "zurück", "zusammen", "zwanzig", "zwar", "zwei", "zweite", "zweiten", "zweiter",
                    "zweites", "zwischen", "zwölf", "euer", "eure", "hattest", "hattet", "jedes", "mußt", "müßt",
                    "sollst", "sollt", "soweit", "weshalb", "wieso", "woher", "wohin", "xa0", "000"]

    text = " ".join(i for i in data)
    stopwords = stopwords_de + list(STOPWORDS)
    wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
    plt.figure( figsize=(15,10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig("wordcloud_head.png", dpi=1800)

    return plt.show()

wordcloud(data_original_refugee["content"])

