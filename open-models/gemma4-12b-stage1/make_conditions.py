# Write the per-condition request fields, including the marker token ids for the Gemma tokenizer.
import json, sys
from tokenizers import Tokenizer
root = sys.argv[1]
tok = Tokenizer.from_file(root + "/planA/tok/tokenizer.json")
surfaces = json.load(open(root + "/planA/markers/marker_token_map.json"))["ids"]
ids, kept = {}, {}
for surface in surfaces:
    word = surface.strip()
    cap = word[0].upper() + word[1:]
    # Gemma writes sentence-initial hedges without a leading space, so cover that form too.
    for form in (" " + word, " " + cap, cap):
        enc = tok.encode(form, add_special_tokens=False).ids
        if len(enc) == 1:
            ids[str(enc[0])] = -2.0
            kept[form] = enc[0]
conds = {"c1": {"temperature": 1.0}, "c2": {"repetition_penalty": 1.08}, "c3": {"logit_bias": ids}, "c4": {}}
# tau2 runs use the leaderboard sampling settings with thinking on, for the base and the distilled model.
agent = {"temperature": 0.6, "top_p": 0.95, "chat_template_kwargs": {"enable_thinking": True}}
conds["tb"], conds["tv"] = dict(agent), dict(agent)
json.dump(conds, open(root + "/stage1/conditions.json", "w"))
json.dump(kept, open(root + "/stage1/marker_forms.json", "w"), ensure_ascii=False, indent=0)
print("marker tokens", len(ids), "forms", list(kept)[:12])
for w in ("Wait", " Wait", " wait", "However", " However", "But", " But", "Hmm", "Actually", "Alternatively"):
    print(repr(w), kept.get(w))
