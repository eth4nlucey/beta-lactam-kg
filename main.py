from scripts.chembl_api import get_targets

drug = "amoxicillin"
print(f"🔍 Fetching targets for: {drug}")
targets = get_targets(drug)

print(f"🎯 ChEMBL targets for {drug}:")
for t in targets:
    print(f" - {t}")
