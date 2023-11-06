import numpy as np
import pandas as pd
import requests
import os
import re
import nltk
from nltk.lm import NgramCounter
from nltk.util import ngrams
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
from sklearn.linear_model import LogisticRegression
from MyCounter import MyCounter

# nltk.download('punkt')
# nltk.download("stopwords")

np.random.seed(10)
url = "https://en.wikipedia.org/w/api.php"
__M_IDS__ = [
    46303046,
    5835531,
    3887850,
    52103535,
    33076141,
    5335383,
    40440492,
    4031180,
    66655182,
    55554061,
    951614,
    66012376,
    10420896,
    3740763,
    3296243,
    6868776,
    38474884,
    300772,
    10616040,
    36732930,
    3089877,
    2947580,
    59397759,
    23714060,
    12123403,
    70605039,
    83877,
    37536553,
    3372919,
    46702056,
    4280266,
    22144213,
    2703534,
    16818556,
    37956283,
    4311925,
    16073320,
    38532630,
    42863440,
    10304186,
    31359857,
    5826435,
    32265516,
    24848067,
    39762849,
    54404726,
    63842993,
    31204957,
    33102664,
    16961703,
    50892477,
    3507309,
    6645369,
    6653275,
    74947786,
    16036651,
    8002789,
    12365985,
    16896801,
    59055215,
    33076955,
    33840728,
    4765289,
    35017862,
    69780644,
    68449046,
    60753035,
    13506609,
    39893187,
    54385959,
    3952518,
    65243878,
    40913990,
    6723049,
    30688232,
    10227263,
    4079053,
    34976715,
    15093896,
    3640605,
    37953948,
    42269169,
    74895888,
    1130479,
    65368595,
    9901652,
    3823715,
    17035364,
    2186220,
    1130546,
    18662881,
    11526568,
    1562717,
    27386995,
    25367107,
    33149349,
    4325643,
    7332373,
    61735633,
    67864558,
    59951553,
    30427095,
    24120957,
    37494278,
    317507,
    62883347,
    69756391,
    526596,
    2201908,
    8155942,
    2618516,
    2329492,
    7703176,
    3976905,
    14747754,
    71872518,
    29252190,
    64649692,
    58157868,
    35667270,
    54414218,
    11721871,
    22826968,
    14892880,
    61233629,
    10275355,
    3837225,
    30748898,
    16344646,
    22635668,
    5697015,
    29830233,
    42649432,
    1130504,
    1130476,
    13436654,
    54317251,
    44487472,
    53248586,
    935486,
    21148859,
    44290674,
    10034950,
    3882906,
    10445988,
    32999836,
    13107678,
    15104461,
    6786575,
    73792936,
    10749768,
    935451,
    35035352,
    6685906,
    40616100,
    9646417,
    72468219,
    22488521,
    26488003,
    36851711,
    68813943,
    70374700,
    9128972,
    5316548,
    7687502,
    1410386,
    41225619,
    25258088,
    55586589,
    33354427,
    67123375,
    44441540,
    52387136,
    35695335,
    13765537,
    3886974,
    26349710,
    11175093,
    41864077,
    26098083,
    33331852,
    17471627,
    61465449,
    4861880,
    21952946,
    27936239,
    56746806,
    11157360,
    43698229,
    12237355,
    61525916,
    32279306,
    66213053,
    27937615,
    5703563,
    49782137,
    49066893,
    919914,
    27232632,
    46223194,
    40703649,
    61015976,
    40615969,
    2439204,
    4870889,
    27942902,
    5551093,
    24032119,
    2410864,
    4097229,
    3498765,
    2201782,
    10612706,
    8625779,
    21291825,
    25530727,
    32850190,
    25336375,
    22613602,
    10110134,
    17495225,
    1532332,
    2165494,
    39963253,
    32812498,
    10324589,
    35539275,
    5846118,
    2015788,
    70379793,
    5143229,
    4565578,
    919125,
    46440709,
    33387123,
    11385740,
    7176424,
    39536693,
    11648613,
    56233051,
    6866234,
    40419859,
    23413270,
    52682951,
    32997558,
    46295506,
    44186297,
    5446800,
    25464216,
    4134885,
    37329385,
    31786777,
    33354565,
    34564790,
    48584298,
    3999078,
    12664718,
    22168869,
    10543241,
    67848628,
    8173473,
    43104215,
    17538021,
    43158726,
    33354679,
    10856772,
    69518370,
    317511,
    3518067,
    8993455,
    52018560,
    33915432,
    63769908,
    284029,
    54393139,
    33138964,
    3902650,
    62792073,
    69627157,
    32854110,
    16040223,
    11789193,
    31881995,
    5312063,
    40616254,
    43565534,
    57491940,
    1410327,
    11064985,
    27387114,
    35539271,
    56378847,
    15219191,
    69150068,
    17608549,
    1607085,
    70379446,
    57371426,
    9298680,
    18659643,
    5484891,
    43051635,
    21862169,
    16106907,
    2622311,
    58516158,
    8156682,
    17542723,
    44301349,
    35539273,
    2167412,
    30873290,
    10139817,
    25755794,
    3972534,
    30031585,
    32241450,
    223111,
    59555366,
    48029696,
    23451032,
    52823792,
    646573,
    29897966,
    13008203,
    19560460,
    56180116,
    28539183,
    69841931,
    1065730,
    11365279,
    19383745,
    10527532,
    11094380,
    40606087,
    58436923,
    32078680,
    38526066,
    63098188,
    10022123,
    2114227,
    56057194,
    16556916,
    1482042,
    23088813,
    33331511,
    8466542,
    33679848,
    7643643,
    39307595,
    10013,
    24426070,
    31591137,
    3466967,
    21977757,
    44884825,
    4871848,
    1005946,
    22314141,
    16386089,
    10066229,
    18090607,
    5669348,
    66893619,
    70827899,
    33373214,
    53068252,
    14820213,
    31347277,
    1370289,
    35834433,
    67445372,
    13041496,
    18660824,
    68505120,
    10405105,
    32473,
    46499719,
    37248915,
    11026468,
    20819509,
    35926498,
    18610161,
    22192759,
    63687773,
    27287411,
    13252823,
    13276327,
    4376466,
    46443694,
    18248420,
    33909875,
    237721,
    47763053,
    39968360,
    39240276,
    9533319,
    19364010,
    39532251,
    16918530,
    1899507,
    6265425,
    71502309,
    5206174,
    55168830,
    15696380,
    32093248,
    1060461,
    22077494,
    6868712,
    58012433,
    2056625,
    17835748,
    3914815,
    18137793,
    2575117,
    46603974,
    25000656,
    32841338,
    2015725,
    71487115,
    87175,
    47393255,
    4098644,
    7324090,
    26777771,
    32165548,
    17719645,
    17323880,
    42448124,
    14883919,
    51452660,
    69563062,
    1424143,
    1692847,
    23847326,
    12173489,
    14414683,
    40542571,
    3914909,
    34306601,
    1281756,
    67757719,
    37910936,
    66338001,
    7247458,
    38088694,
    58011203,
    65367112,
    8002693,
    16067905,
    1130490,
    30540048,
    46864621,
    34677623,
    40690322,
    4288821,
]

__NON_M_IDS__ = [
    7410249,
    18400571,
    90138,
    5571005,
    53290497,
    5832437,
    48110,
    1840762,
    2958015,
    8099572,
    18472072,
    4513331,
    25508360,
    3054853,
    68092158,
    34043,
    440393,
    36082813,
    4175228,
    13666328,
    64465154,
    55762330,
    19810565,
    297267,
    47123596,
    58956661,
    48043351,
    71698731,
    24818668,
    53034676,
    118474,
    18836,
    74185828,
    20756311,
    70991052,
    1742461,
    13878013,
    6460996,
    72166473,
    61593115,
    16707204,
    6420600,
    43740334,
    68027374,
    47243509,
    37679667,
    3091760,
    2886641,
    27812753,
    59329248,
    50345901,
    5904871,
    39563457,
    592392,
    7183413,
    20395887,
    67654734,
    69833142,
    18345,
    27290438,
    44022067,
    3069483,
    97234,
    69822725,
    49505197,
    2234574,
    48069054,
    53580439,
    852480,
    6323098,
    426484,
    722906,
    53880545,
    1975092,
    8961575,
    1851712,
    4522868,
    33034640,
    62433972,
    68359750,
    17046772,
    25495107,
    9755539,
    21259396,
    44636570,
    37679646,
    64688191,
    2548050,
    516138,
    55941593,
    27314675,
    74911613,
    430976,
    49073376,
    14173752,
    3234630,
    2694761,
    45498419,
    46181931,
    50943294,
    36807165,
    3872234,
    52017909,
    6824237,
    57837450,
    5822209,
    8919856,
    70093812,
    62996247,
    56221934,
    41952735,
    69266674,
    54245,
    41334623,
    24991288,
    70913289,
    57192745,
    8245348,
    33252958,
    73499130,
    36979916,
    22156167,
    1419427,
    9332507,
    498971,
    25776777,
    57512513,
    29570293,
    73761282,
    1478662,
    20871496,
    40914232,
    16300571,
    64041906,
    55817338,
    32172794,
    30792823,
    188168,
    363092,
    470031,
    16189184,
    25590689,
    39127991,
    67525526,
    21666977,
    646598,
    989968,
    75081155,
    11977904,
    17611421,
    8994982,
    34956958,
    5823212,
    351887,
    181117,
    372984,
    69600247,
    54116315,
    47123594,
    24503316,
    51472964,
    40182149,
    58455529,
    12719721,
    848317,
    62521794,
    66763727,
    59969032,
    7251223,
    433005,
    74558667,
    42431426,
    66401997,
    72545591,
    72135653,
    12454750,
    1001254,
    12476035,
    63192624,
    30405742,
    993125,
    956795,
    24587014,
    1244693,
    2685999,
    66188994,
    722374,
    70575453,
    39728112,
    46673516,
    58221338,
    2110568,
    44007076,
    62712050,
    30858332,
    2968419,
    71759612,
    48120778,
    58247763,
    39748081,
    72772900,
    24616694,
    31896902,
    350381,
    61588611,
    30299,
    23569174,
    27887526,
    13557443,
    404048,
    55371131,
    64768088,
    1124646,
    2119393,
    63224981,
    3547088,
    33893690,
    23529,
    53004548,
    12749679,
    1117386,
    70126407,
    66704693,
    8191076,
    55571473,
    28244920,
    32923,
    1690046,
    14248637,
    21391751,
    21279463,
    74958182,
    53561288,
    32237314,
    33957251,
    983787,
    32544555,
    70174,
    103155,
    7174,
    8010462,
    242315,
    51781199,
    39403716,
    465480,
    75030221,
    1523896,
    263636,
    13659583,
    27705110,
    8434470,
    69276509,
    13607556,
    69460867,
    1134562,
    44208525,
    21148945,
    43401246,
    53977963,
    65029937,
    51676196,
    20959122,
    21424551,
    66205491,
    4943880,
    60557698,
    62191085,
    62683332,
    18963870,
    70476313,
    16677153,
    4137340,
    52627686,
    38024285,
    2588267,
    62971006,
    37083057,
    1191544,
    36306767,
    46751769,
    10938364,
    71498836,
    8443072,
    3837845,
    2712053,
    455547,
    50785023,
    56078430,
    39895391,
    5728552,
    69070022,
    28381078,
    17709742,
    291229,
    718763,
    42429912,
    70993391,
    41071610,
    20593859,
    8307635,
    21981132,
    50700186,
    54840463,
    4218673,
    37389994,
    532476,
    62483077,
    70018114,
    952926,
    14343000,
    562583,
    22926,
    61481498,
    46419264,
    4127809,
    5808948,
    55146723,
    74273322,
    404037,
    621946,
    78534,
    208999,
    25172837,
    57491343,
    14842794,
    67006520,
    50741246,
    4647646,
    18801301,
    55962927,
    3037867,
    24145205,
    47937215,
    15944015,
    2200600,
    340510,
    31240382,
    10088016,
    27667,
    46382079,
    649720,
    48215783,
    52059694,
    31641770,
    25136,
    47484119,
    30999254,
    28726646,
    62902519,
    40056381,
    10276033,
    70673979,
    22141996,
    72682229,
    6273713,
    43692456,
    33563052,
    19284595,
    73826791,
    63631356,
    26184869,
    51461892,
    68845945,
    3951220,
    27691317,
    17569583,
    4444344,
    3536209,
    57352318,
    25046128,
    43417461,
    951649,
    73501753,
    72607666,
    24220268,
    12207,
    8856044,
    62489620,
    29997493,
    54072382,
    302445,
    42429675,
    27425975,
    13333913,
    19257688,
    22805820,
    11595620,
    62980659,
    69864289,
    3328588,
    2909033,
    898161,
    6216,
    29663719,
    37043764,
    1123773,
    27606636,
    10967568,
    33739475,
]

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


def logistic_regressor(validation=False):
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
        __validation_data__.drop("id", inplace=True, axis=1)

        logistic = LogisticRegression("l2")
        fitted = logistic.fit(X_training, y_training)

        yhat = fitted.predict(X_validation)

        precision = round(precision_score(y_validation, yhat), 3)
        recall = round(recall_score(y_validation, yhat), 3)
        f1 = round(f1_score(y_validation, yhat), 3)
        print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")
    else:
        logistic = LogisticRegression("l2")
        fitted = logistic.fit(X_training, y_training)

        yhat = fitted.predict(X_test)

        precision = round(precision_score(y_test, yhat), 3)
        recall = round(recall_score(y_test, yhat), 3)
        f1 = round(f1_score(y_test, yhat), 3)
        print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")


# naive_bayes()
logistic_regressor(validation=True)
