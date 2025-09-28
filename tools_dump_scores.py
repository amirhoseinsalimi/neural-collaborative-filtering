import argparse, json, os
import numpy as np

from keras import backend as K
from NeuMF import get_model

def infer_num_users_items(dataset_name, base_path="Data"):
    try:
        from Dataset import Dataset
        ds = Dataset(os.path.join(base_path, dataset_name))
        for u_attr in ["num_users", "userNum", "n_users", "user_count"]:
            for i_attr in ["num_items", "itemNum", "n_items", "item_count"]:
                if hasattr(ds, u_attr) and hasattr(ds, i_attr):
                    return int(getattr(ds, u_attr)), int(getattr(ds, i_attr))
        if hasattr(ds, "trainMatrix") and hasattr(ds.trainMatrix, "shape"):
            return int(ds.trainMatrix.shape[0]), int(ds.trainMatrix.shape[1])
    except Exception:
        pass

    train_path = os.path.join(base_path, "%s.train.rating" % dataset_name)
    max_u, max_i = -1, -1
    with open(train_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                parts = line.strip().split(" ")
            if len(parts) >= 2:
                try:
                    u = int(parts[0]); i = int(parts[1])
                    if u > max_u: max_u = u
                    if i > max_i: max_i = i
                except:
                    continue
    return max_u + 1, max_i + 1

def batched_predict(model, user_ids, item_ids, batch=4096):
    out = []
    N = len(user_ids)
    for i in range(0, N, batch):
        u = np.asarray(user_ids[i:i+batch], dtype='int32')
        it = np.asarray(item_ids[i:i+batch], dtype='int32')
        s = model.predict([u, it], batch_size=min(batch, len(u)), verbose=0).reshape(-1)
        out.append(s)
    return np.concatenate(out, axis=0)

def parse_list(s):
    s = s.strip().strip('[]')
    return [int(x.strip()) for x in s.split(',') if x.strip()]

def build_neumf(num_users, num_items, layers, reg_layers, num_factors, reg_mf):
    import inspect

    all_args = []
    try:
        # Python 2
        spec = inspect.getargspec(get_model)
        all_args = list(spec.args or [])
    except Exception:
        pass

    # Map flexible kw names
    kw = {}
    if 'layers' in all_args: kw['layers'] = layers
    if 'reg_layers' in all_args: kw['reg_layers'] = reg_layers
    if 'num_factors' in all_args: kw['num_factors'] = num_factors
    if 'mf_dim' in all_args: kw['mf_dim'] = num_factors  # same meaning
    if 'reg_mf' in all_args: kw['reg_mf'] = reg_mf

    return get_model(num_users, num_items, **kw)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True, help="weights .h5 path inside container")
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_factors", type=int, default=8)
    ap.add_argument("--layers", type=str, default="[64,32,16,8]")
    ap.add_argument("--reg_layers", type=str, default="[0,0,0,0]")
    ap.add_argument("--reg_mf", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=4096)
    args = ap.parse_args()

    layers = parse_list(args.layers)
    reg_layers = [float(x) for x in parse_list(args.reg_layers)]

    num_users, num_items = infer_num_users_items(args.dataset, base_path="Data")

    model = build_neumf(num_users, num_items,
                        layers=layers,
                        reg_layers=reg_layers,
                        num_factors=args.num_factors,
                        reg_mf=args.reg_mf)

    # Load weights (Keras 1.x style, weights-only .h5)
    model.load_weights(args.model)

    with open(args.candidates) as fin, open(args.out, "w") as fout:
        for line in fin:
            obj = json.loads(line)
            u = int(obj["user_id"])
            items = [int(obj["pos_item_id"])] + [int(x) for x in obj["neg_item_ids"]]
            users = [u] * len(items)
            scores = batched_predict(model, users, items, batch=args.batch).tolist()
            fout.write(json.dumps({"user_id": u, "items": items, "scores": scores}) + "\n")

if __name__ == "__main__":
    main()
