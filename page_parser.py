import requests

def get_ids(url, categories):
    returned_ids = []

    for c in categories:
        params = {
            "action": "query",
            "cmtitle": c,
            "cmlimit": "500",
            # "cmtype": "subcat",
            "list": "categorymembers",
            "format": "json",
        }

        req = requests.get(url=url, params=params)
        pages = req.json()["query"]["categorymembers"]

        page_ids = [page["pageid"] for page in pages]
        print(len(page_ids))

        for id in page_ids:
            new_params = {
                "format": "json",
                "action": "query",
                "prop": "extracts",
                "exintro": True,
                "explaintext": True,
                "redirects": 1,
                "pageids": id,
            }
            req = requests.get(url, new_params)
            try:
                title = req.json()["query"]["pages"][str(id)]["title"]
                # print(title)
                if (
                    title.startswith("Category")
                    or title.startswith("Template")
                    or title.startswith("Portal")
                ):
                    continue
                else:
                    returned_ids.append(id)
            except:
                print(f"||Failed at id {id}||")

    return returned_ids


url = "https://en.wikipedia.org/w/api.php"

medical_categories = [
    "Category:Alternative medicine stubs",
    "Category:Evidence-based medicine",
    "Category:Veterinary medicine stubs",
    "Category:Vaccination",
    "Category:2018 disease outbreaks",
]

non_medical_categories = [
    "Category:Dark ages",
    "Category:Historiography of China",
    "Category:Sports controversies",
    "Category:Philosophy of artificial intelligence",
    "Category:Geology",
    "Category:Space",
    "Category:Literature",
    "Category:Music videos",
]


print(get_ids(url, non_medical_categories))
