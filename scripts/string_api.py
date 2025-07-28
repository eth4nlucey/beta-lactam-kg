import requests

def get_string_interactions(identifiers, species=9606):
    url = "https://string-db.org/api/tsv/network"
    params = {
        "identifiers": "%0A".join(identifiers),
        "species": species
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.text
    else:
        print("Error:", response.status_code)
        return None
