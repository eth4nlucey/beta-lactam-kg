import requests

def get_target_metadata(chembl_target_id):
    """
    Attempts to retrieve UniProt accessions for a ChEMBL target ID using the /target endpoint.
    Works for some targets, but limited for bacterial targets like antibiotics.
    """
    url = f"https://www.ebi.ac.uk/chembl/api/data/target/{chembl_target_id}.json"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            target_components = data.get("target_components", [])
            accessions = []

            for comp in target_components:
                for obj in comp.get("target_components", []):
                    acc = obj.get("accession")
                    if acc:
                        accessions.append(acc)

            return accessions if accessions else ["⚠️ No UniProt accession found"]
        else:
            print(f"❌ Failed to fetch metadata for {chembl_target_id}: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Error for {chembl_target_id}: {e}")
        return []


def get_mechanism_targets(drug_chembl_id):
    """
    Retrieves UniProt accessions via ChEMBL's /mechanism endpoint, 
    which includes known mechanisms of action for a drug.
    This is more reliable for antibiotics and non-human targets.
    """
    url = f"https://www.ebi.ac.uk/chembl/api/data/mechanism.json?molecule_chembl_id={drug_chembl_id}"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            mechanisms = data.get("mechanisms", [])
            accessions = []

            for mech in mechanisms:
                components = mech.get("target_components", [])
                for comp in components:
                    acc = comp.get("accession")
                    if acc:
                        accessions.append(acc)

            return list(set(accessions)) if accessions else ["⚠️ No UniProt accession found"]
        else:
            print(f"❌ Failed to fetch mechanism data: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Error during mechanism lookup: {e}")
        return []
