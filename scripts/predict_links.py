import torch
from pykeen.triples import TriplesFactory

def predict_tail_entities(model_path, triples_path, head, relation, candidates):
    tf = TriplesFactory.from_path(triples_path)
    model = torch.load(model_path, weights_only=False).to("cpu").eval()

    # Get entity and relation ID mappings
    entity_to_id = tf.entity_to_id
    relation_to_id = tf.relation_to_id

    h_id = entity_to_id[head]
    r_id = relation_to_id[relation]

    scores = []
    for tail in candidates:
        t_id = entity_to_id[tail]
        hrt_tensor = torch.tensor([[h_id, r_id, t_id]])
        score = model.score_hrt(hrt_tensor).item()
        scores.append((tail, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
