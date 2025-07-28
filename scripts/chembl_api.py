import requests

def get_targets(drug_name):
    print(f"🔍 Searching for ChEMBL compound: {drug_name}")

    # Step 1: Search for ChEMBL ID
    search_url = "https://www.ebi.ac.uk/chembl/api/data/molecule"
    search_params = {"format": "json", "q": drug_name}
    search_response = requests.get(search_url, params=search_params, timeout=10)

    if search_response.status_code != 200:
        print("❌ ChEMBL search failed")
        return []

    try:
        molecules = search_response.json().get("molecules", [])
        if not molecules:
            print("⚠️ No molecule found.")
            return []
        
        chembl_id = molecules[0].get("molecule_chembl_id")
        print(f"🔗 Found ChEMBL ID: {chembl_id}")
    except Exception as e:
        print(f"❌ JSON error during ChEMBL ID lookup: {e}")
        return []

    # Step 2: Fetch target info
    target_url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json"
    target_params = {
        "molecule_chembl_id": chembl_id,
        "limit": 1000
    }
    target_response = requests.get(target_url, params=target_params, timeout=10)

    if target_response.status_code != 200:
        print("❌ ChEMBL target fetch failed")
        return []

    try:
        activities = target_response.json().get("activities", [])
        targets = set()

        for activity in activities:
            target = activity.get("target_chembl_id")
            if target:
                targets.add(target)

        return list(targets)
    except Exception as e:
        print(f"❌ JSON error during activity fetch: {e}")
        return []


