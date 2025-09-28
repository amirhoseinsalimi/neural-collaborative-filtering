import argparse, json
import numpy as np
from tensorflow.keras.models import load_model

def batched_predict(model, user_ids, item_ids, batch=4096):
    out = []; N = len(user_ids)
    for i in range(0, N, batch):
        u  = np.asarray(user_ids[i:i+batch], dtype='int32')
        it = np.asarray(item_ids[i:i+batch], dtype='int32')
        s  = model.predict([u, it], batch_size=min(batch, len(u)), verbose=0).reshape(-1)
        out.append(s)
    return np.concatenate(out, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=4096)
    args = ap.parse_args()

    model = load_model(args.model, compile=False)
    with open(args.candidates) as fin, open(args.out, "w") as fout:
        for line in fin:
            obj   = json.loads(line)
            u     = int(obj["user_id"])
            items = [int(obj["pos_item_id"])] + [int(x) for x in obj["neg_item_ids"]]
            users = [u] * len(items)
            scores = batched_predict(model, users, items, batch=args.batch).tolist()
            fout.write(json.dumps({"user_id": u, "items": items, "scores": scores}) + "\n")

if __name__ == "__main__":
    main()
