import requests

def get_string_interactions(identifiers, species=511145):
    """
    Given a list of gene/protein identifiers and a species ID,
    returns STRING interaction data in TSV format.
    """
    url = "https://string-db.org/api/tsv/network"
    params = {
        "identifiers": "%0A".join(identifiers),
        "species": species
    }
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        return response.text
    else:
        print(f"❌ STRING API error {response.status_code}")
        print("Raw response:")
        print(response.text)
        return None
