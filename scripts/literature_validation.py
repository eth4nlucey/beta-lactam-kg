import requests
import time

def search_literature_evidence(drug1, drug2, max_results=5):
    """Search Europe PMC for literature evidence of drug combinations"""
    
    # Construct search query
    query = f'"{drug1}" AND "{drug2}" AND ("synergy" OR "combination" OR "adjuvant")'
    
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {
        "query": query,
        "format": "json",
        "resultType": "core",
        "pageSize": max_results
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            articles = data.get("resultList", {}).get("result", [])
            
            evidence = []
            for article in articles:
                title = article.get("title", "No title")
                pmid = article.get("pmid", "No PMID")
                evidence.append({"title": title, "pmid": pmid})
            
            return evidence
        else:
            print(f"❌ Europe PMC error: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"❌ Literature search error: {e}")
        return []

def validate_top_predictions(predictions, max_combinations=3):
    """Validate top adjuvant predictions against literature"""
    
    print("📚 Validating predictions against literature...")
    validated_results = []
    
    for i, (drug, score, shared_targets) in enumerate(predictions[:max_combinations]):
        print(f"\n🔍 Searching literature for: Ampicillin + {drug}")
        
        evidence = search_literature_evidence("Ampicillin", drug)
        
        result = {
            "combination": f"Ampicillin + {drug}",
            "prediction_score": score,
            "shared_targets": len(shared_targets),
            "literature_evidence": len(evidence),
            "articles": evidence
        }
        
        validated_results.append(result)
        
        # Be nice to the API
        time.sleep(1)
    
    return validated_results