import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import os
import re
import nltk
from nltk.lm import NgramCounter
from nltk.util import ngrams
from collections import Counter
from sklearn.metrics import RocCurveDisplay, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.linear_model import LogisticRegression
from MyCounter import MyCounter
from ids import __M_IDS__, __NON_M_IDS__

# nltk.download('punkt')
# nltk.download("stopwords")
nltk.download("wordnet")

np.random.seed(10)
url = "https://en.wikipedia.org/w/api.php"

VALIDATION_SIZE = 86  # 10% of the dataset
print(
    f"Medical dataset size: {len(__M_IDS__)}, Non medical dataset size: {len(__NON_M_IDS__)}"
)


def get_training_ids():
    ratio = len(__M_IDS__) / (len(__M_IDS__) + len(__NON_M_IDS__))

    quote_m = int(ratio * VALIDATION_SIZE)
    quote_n_m = VALIDATION_SIZE - quote_m

    training_m_ids = __M_IDS__[quote_m:]
    training_non_m_ids = __NON_M_IDS__[quote_n_m:]
    return training_m_ids, training_non_m_ids


def get_validation_ids():
    ratio = len(__M_IDS__) / (len(__M_IDS__) + len(__NON_M_IDS__))

    quote_m = int(ratio * VALIDATION_SIZE)
    quote_n_m = VALIDATION_SIZE - quote_m

    validation_m_ids = __M_IDS__[0:quote_m]
    validation_non_m_ids = __NON_M_IDS__[0:quote_n_m]
    return validation_m_ids, validation_non_m_ids


def get_train_test(ids, p):
    np.random.shuffle(ids)
    quote = int(len(ids) * p)
    return ids[:quote], ids[quote:]


def make_labels(medical_ids, non_medical_ids):
    labels = {}
    for mi in medical_ids:
        labels[mi] = 1
    for nmi in non_medical_ids:
        labels[nmi] = 0
    return labels


def produce_documents(ids, kind):
    for id in ids:
        new_params = {
            "format": "json",
            "action": "query",
            "prop": "revisions",
            "rvslots": "*",
            "rvprop": "content",
            "redirects": 1,
            "pageids": id,
        }
        req = requests.get(url, new_params)
        try:
            title = req.json()["query"]["pages"][str(id)]["revisions"][0]["slots"][
                "main"
            ]["*"]
            with open(f"./documents/{kind}/{id}.txt", "w") as f:
                f.write(title)
        except:
            print(f"||Failed at id {id}||")


def clean_doc(string):
    string = re.sub("<ref.*?</ref>", "", string)  # removes refs
    string = re.sub("<ref.*?/>", "", string)  # idem
    string = re.sub("{.*?}", "", string)  # removes "{...}"
    string = re.sub(
        "\|.*\n?", "", string
    )  # removes lines starting with "|"" and continuing until the end
    string = re.sub("(Category).*\n?", "", string)
    string = re.sub("(thumb\|.*?\|)", "", string)  # removes "thumb|...|"
    string = re.sub(
        "(thumb)", "", string
    )  # removes "thumb" (canno easily distinguish all cases)
    string = re.sub(
        "\[\[.*?\|", "", string
    )  # removes links such as [[dieting|diet]], but only the first part (up until "|"), which is the link.
    string = re.sub(
        "[\[,\],{,},',\\',\,\.,#,=,*\|`-]", "", string
    )  # removes all remaining bad characters: left out [], {}, #, =, |, ', `, -, *
    string = re.sub("\\n", "\n", string)  # removes newlines
    return string


def clean_documents(folder):
    path = f"./documents/{folder}"
    os.chdir(path)
    for file in os.listdir():
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            file_path = f"{path}/{file}"

            # call read text file function
            print(file_path)
            new_lines = []
            with open(file, "r") as f:
                lines = f.readlines()
                for l in lines:
                    new_lines.append(clean_doc(l))
            f_path = file.split(".")[0]
            new_path = f"../{f_path}_c.txt"
            with open(new_path, "w") as fw:
                for nl in new_lines:
                    fw.write(nl)


def tokenize(text):
    return nltk.word_tokenize(text)


def make_bow(id):
    path = f"./documents/{id}_c.txt"
    with open(path, "r") as f:
        tokens = []

        lines = f.readlines()
        for l in lines:
            tok = tokenize(l)
            tokens = tokens + tok

        # print(tokens)
        counts = MyCounter(tokens)

        counts2 = counts.remove_stopwords()

        counts3 = counts2.stem()
        return counts3


def predict_and_score(ids, labels, priors, m_doc, nm_doc):
    yhat = []
    y = []
    for i in ids:
        likelihoods = [
            make_bow(i).log_likelihood_document(m_doc),
            make_bow(i).log_likelihood_document(nm_doc),
        ]

        posteriors = []

        for j in range(len(likelihoods)):
            posteriors.append(likelihoods[j] + np.log(priors[j]))

        yhat.append(1 if posteriors[0] >= posteriors[1] else 0)
        y.append(labels[i])

    precision = round(precision_score(y, yhat), 3)
    recall = round(recall_score(y, yhat), 3)
    f1 = round(f1_score(y, yhat), 3)
    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")


def naive_bayes():
    labels = make_labels(__M_IDS__, __NON_M_IDS__)
    medical_training_set, non_medical_training_set = get_training_ids()
    medical_validation_set, non_medical_validation_set = get_validation_ids()

    assert len(medical_training_set) + len(medical_validation_set) == len(__M_IDS__)
    assert len(non_medical_training_set) + len(non_medical_validation_set) == len(
        __NON_M_IDS__
    )

    # print(labels)

    # produce_documents(__NON_M_IDS__, "non_medicine")
    #

    medical_mega_document = MyCounter({}, stemmed=True)

    for mi in medical_training_set:
        medical_mega_document.update(make_bow(mi))

    non_medical_mega_document = MyCounter({}, stemmed=True)

    for nmi in non_medical_training_set:
        non_medical_mega_document.update(make_bow(nmi))

    ratio = len(medical_training_set) / (
        len(medical_training_set) + len(non_medical_training_set)
    )
    priors = [ratio, 1 - ratio]
    validation_set = medical_validation_set + non_medical_validation_set

    predict_and_score(
        validation_set, labels, priors, medical_mega_document, non_medical_mega_document
    )


# clean_documents("medicine")
# clean_documents("non_medicine")


def make_dataset(kind, ids, labels, most_common):
    np.random.shuffle(ids)
    columns = [
        "id",
        "c",
    ] + [str(n) for n in range(len(most_common))]
    dataset = []

    for id in ids:
        c = labels[id]
        features = [id, c]
        bow = make_bow(id)

        for word in most_common:
            if bow.contains(word):
                features.append(1)
            else:
                features.append(0)

        dataset.append(features)

    df = pd.DataFrame(dataset, columns=columns)
    # print(df)
    df.to_csv(f"./{kind}_dataset.csv", index=False)


def logistic_regressor(validation=False, plot=False):
    labels = make_labels(__M_IDS__, __NON_M_IDS__)
    # Set to be used as training and testing
    medical_set, non_medical_set = get_training_ids()
    # Validation set (double underscore because private, not the be used until the end)
    __medical_validation_set__, __non_medical_validation_set__ = get_validation_ids()

    medical_training_set, medical_test_set = get_train_test(medical_set, 0.75)
    non_medical_training_set, non_medical_test_set = get_train_test(
        non_medical_set, 0.75
    )

    # print(f"Medical: {len(medical_training_set)} + {len(medical_test_set)}")
    # print(f"Non Medical: {len(non_medical_training_set)} + {len(non_medical_test_set)}")

    # Create mega document from training set only, not test set or validation set
    medical_mega_document = MyCounter({}, stemmed=True)
    for mi in medical_training_set:
        medical_mega_document.update(make_bow(mi))

    non_medical_mega_document = MyCounter({}, stemmed=True)
    for nmi in non_medical_training_set:
        non_medical_mega_document.update(make_bow(nmi))

    m_common = [a[0] for a in medical_mega_document.most_common(150)]
    non_m_common = [a[0] for a in non_medical_mega_document.most_common(150)]

    common = m_common + non_m_common
    """
    make_dataset(
        "training", medical_training_set + non_medical_training_set, labels, common
    )
    make_dataset("test", medical_test_set + non_medical_test_set, labels, common)
    make_dataset(
        "validation",
        __medical_validation_set__ + __non_medical_validation_set__,
        labels,
        common,
    )
    """
    # print(common)
    training_data = pd.read_csv("./training_dataset.csv")
    test_data = pd.read_csv("./test_dataset.csv")
    __validation_data__ = pd.read_csv("./validation_dataset.csv")

    y_training = training_data.pop("c")
    X_training = training_data
    X_training.drop("id", inplace=True, axis=1)

    y_test = test_data.pop("c")
    X_test = test_data
    X_test.drop("id", inplace=True, axis=1)

    if validation:
        print("== Validation set ==")
        # Train on training + test and predict validation set
        X_training = pd.concat([X_training, X_test])
        y_training = pd.concat([y_training, y_test])

        y_validation = __validation_data__.pop("c")
        X_validation = __validation_data__
        X_validation.drop("id", inplace=True, axis=1)

        logistic = LogisticRegression("l2")
        fitted = logistic.fit(X_training, y_training)

        yhat = fitted.predict(X_validation)
        y_score = fitted.predict_proba(X_test)
        precision = round(precision_score(y_validation, yhat), 3)
        recall = round(recall_score(y_validation, yhat), 3)
        f1 = round(f1_score(y_validation, yhat), 3)
        print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")
        if plot:
            RocCurveDisplay.from_estimator(
                fitted, X_validation, y_validation, plot_chance_level=True
            )

            plt.axis("square")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("One-vs-Rest ROC curves: Medical vs Non-Medical")
            plt.legend()
            plt.show()
    else:
        logistic = LogisticRegression("l2")
        fitted = logistic.fit(X_training, y_training)

        yhat = fitted.predict(X_test)
        y_score = fitted.predict_proba(X_test)
        proba = fitted.predict_proba(X_test)
        # print(proba)
        precision = round(precision_score(y_test, yhat), 3)
        recall = round(recall_score(y_test, yhat), 3)
        f1 = round(f1_score(y_test, yhat), 3)
        print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")
        if plot:
            RocCurveDisplay.from_estimator(
                fitted, X_test, y_test, plot_chance_level=True
            ),

            plt.axis("square")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("One-vs-Rest ROC curves: Medical vs Non-Medical")
            plt.legend()
            plt.show()


# naive_bayes()
logistic_regressor(validation=True, plot=True)
