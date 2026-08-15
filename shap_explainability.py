
#PhishGuard-GH — SHAP GradientExplainer


import os, time, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
import keras
import shap
import json

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.layers import (Input, Conv1D, MaxPooling1D, Bidirectional,
                          LSTM, Dense, Dropout, Concatenate)

print("SHAP GradientExplainer")

# Configuration
N_EXPLAIN    = 200
N_BACKGROUND = 200
MAX_LEN      = 200

# Character decoding table (inverse of CHAR_TO_INT in train_hash.py)
int_to_chr = {i - 31: chr(i) for i in range(32, 127)}


def decode_seq(seq):
    """Turn an integer-encoded sequence back into its URL string."""
    return ''.join(int_to_chr.get(int(c), '') for c in seq if int(c) != 0)


def is_ip_host(url):
    """True if the URL's hostname is a bare IPv4 literal (scheme already stripped)."""
    host = url.split('/')[0].split(':')[0]
    return bool(re.fullmatch(r'\d{1,3}(\.\d{1,3}){3}', host))


# Load saved files
print("\n Loading saved files")
X_test     = np.load("X_test.npy")
y_test     = np.load("y_test.npy")
background = np.load("X_train_background.npy")[:N_BACKGROUND]
y_prob     = np.load("y_prob.npy")
y_prob_sample = None  # assigned after balancing below
print(f"    X_test shape     : {X_test.shape}")
print(f"    Background shape : {background.shape}")

# Load original model
print("\n Loading model")
if os.path.exists("phishguard_gh_model.keras"):
    original = keras.models.load_model("phishguard_gh_model.keras", compile=False)
else:
    original = keras.models.load_model("phishguard_gh_model.h5", compile=False)
print(f"    Model input shape: {original.input_shape}")

emb_layer = original.get_layer("char_embedding")
conv_k3   = original.get_layer("conv_k3")
conv_k5   = original.get_layer("conv_k5")
maxpool   = original.get_layer("maxpool")
bilstm    = original.get_layer("bilstm")
dense64   = original.get_layer("dense64")
out_layer = original.get_layer("output")

EMBED_DIM = emb_layer.output_dim
print(f"    EMBED_DIM={EMBED_DIM}, MAX_LEN={MAX_LEN}")

# Build float-input model for SHAP
print("\nBuilding float input model for SHAP")
emb_input = Input(shape=(MAX_LEN, EMBED_DIM), name="emb_input", dtype="float32")
c3  = Conv1D(conv_k3.filters, conv_k3.kernel_size[0],
             activation="relu", padding="same", name="f_conv_k3")(emb_input)
c5  = Conv1D(conv_k5.filters, conv_k5.kernel_size[0],
             activation="relu", padding="same", name="f_conv_k5")(emb_input)
x   = Concatenate(name="f_merge")([c3, c5])
x   = MaxPooling1D(pool_size=maxpool.pool_size[0], name="f_maxpool")(x)
# LSTM units read from the trained model so this stays correct if the
# architecture is retuned again (currently 64).
_lstm_units = bilstm.forward_layer.units
x   = Bidirectional(LSTM(_lstm_units), name="f_bilstm")(x)
x   = Dropout(0.5, name="f_dropout")(x)
x   = Dense(dense64.units, activation="relu", name="f_dense64")(x)
out = Dense(1, activation="sigmoid", name="f_output")(x)
float_model = keras.Model(inputs=emb_input, outputs=out, name="PhishGuard_Float")

float_model.get_layer("f_conv_k3").set_weights(conv_k3.get_weights())
float_model.get_layer("f_conv_k5").set_weights(conv_k5.get_weights())
float_model.get_layer("f_bilstm").set_weights(bilstm.get_weights())
float_model.get_layer("f_dense64").set_weights(dense64.get_weights())
float_model.get_layer("f_output").set_weights(out_layer.get_weights())
print(f"    Float model built (BiLSTM units = {_lstm_units}) and weights copied")

# Verify float model matches original
emb_fn   = original.get_layer("char_embedding")
sample5  = X_test[:5].astype(np.int32)
emb5     = emb_fn(tf.constant(sample5, dtype=tf.int32)).numpy()
orig_p   = original.predict(X_test[:5].astype(np.float32), verbose=0).flatten()
float_p  = float_model.predict(emb5, verbose=0).flatten()
max_diff = float(np.max(np.abs(orig_p - float_p)))
print(f"    Prediction match check (should be ~0): {max_diff:.8f}")
print("    Float model verified correctly" if max_diff < 0.01
      else "    WARNING: difference > 0.01 — check weight copying")

# Prepare SHAP inputs
print("\nPreparing SHAP inputs...")
half = N_EXPLAIN // 2
phish_idx = np.where(y_test == 1)[0][:half]
legit_idx = np.where(y_test == 0)[0][:half]
balanced_idx = np.concatenate([phish_idx, legit_idx])
np.random.seed(42)
np.random.shuffle(balanced_idx)

X_sample = X_test[balanced_idx].astype(np.int32)
y_sample = y_test[balanced_idx]
bg_ints  = background[:N_BACKGROUND].astype(np.int32)
y_prob_sample = y_prob[balanced_idx]
print(f"    Balanced sample: {(y_sample==1).sum()} phishing, {(y_sample==0).sum()} legitimate")

X_emb_explain = emb_fn(tf.constant(X_sample, dtype=tf.int32)).numpy().astype(np.float32)
X_emb_bg      = emb_fn(tf.constant(bg_ints,  dtype=tf.int32)).numpy().astype(np.float32)
print(f"    Explain input shape : {X_emb_explain.shape}")
print(f"    Background shape    : {X_emb_bg.shape}")

# Run SHAP
print(f"\n Running SHAP GradientExplainer...")
print(f"    Background: {N_BACKGROUND} samples")
print(f"    URLs to explain: {N_EXPLAIN}\n")

explainer = shap.GradientExplainer(float_model, X_emb_bg)
shap_values_list = []
start = time.time()

for i in range(N_EXPLAIN):
    single_emb = X_emb_explain[i:i+1]
    sv = explainer.shap_values(single_emb)
    if isinstance(sv, list):
        sv = sv[0]
    sv_collapsed = np.asarray(sv)[0]
    while sv_collapsed.ndim > 1:              # collapse embedding dims -> (200,)
        sv_collapsed = sv_collapsed.sum(axis=-1)
    shap_values_list.append(sv_collapsed)

    elapsed   = (time.time() - start) / 60
    remaining = elapsed / (i + 1) * (N_EXPLAIN - i - 1)
    print(f"    URL {i+1:3d}/{N_EXPLAIN} | elapsed {elapsed:.1f}m | remaining {remaining:.1f}m")

shap_values = np.array(shap_values_list)      # (N_EXPLAIN, 200)
np.save("shap_values.npy", shap_values)
print(f"\n    SHAP complete. Values shape: {shap_values.shape}")
print(f"    Saved: shap_values.npy")

# Summary chart
print("\nSaving SHAP summary chart")
sv_2d = shap_values
while sv_2d.ndim > 2:
    sv_2d = sv_2d.sum(axis=-1)
print(f"    shap_values shape after collapse: {sv_2d.shape}")

mean_abs  = np.abs(sv_2d).mean(axis=0).flatten()
top20_idx = np.argsort(mean_abs)[-20:][::-1]
bar_data  = mean_abs[top20_idx][::-1].tolist()
mean_val  = float(mean_abs.mean())

fig, ax = plt.subplots(figsize=(10, 7))
bar_colors = [plt.cm.Reds(float(v)) for v in np.linspace(0.4, 0.9, 20).tolist()]
ax.barh(list(range(20)), bar_data, color=bar_colors[::-1],
        edgecolor="white", linewidth=0.5)
ax.set_yticks(list(range(20)))
ax.set_yticklabels([f"Position {i}" for i in top20_idx[::-1]], fontsize=10)
ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
ax.set_title("Top 20 Most Influential URL Character Positions\n"
             "PhishGuard-GH — SHAP GradientExplainer (Lundberg & Lee, 2017)",
             fontsize=13, fontweight="bold")
ax.axvline(mean_val, color="steelblue", linestyle="--",
           linewidth=1.5, label=f"Mean: {mean_val:.5f}")
ax.legend(fontsize=10)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved: shap_summary.png")

# Print the table the thesis needs (Table 4.5)
print(f"\n  GLOBAL ATTRIBUTION SUMMARY (for Table 4.5)")
print(f"    Global mean |SHAP| across all 200 positions: {mean_val:.5f}")
print(f"    Top 5 positions:")
for r, p in enumerate(top20_idx[:5], 1):
    print(f"      Rank {r}: Position {p:3d} | mean |SHAP| = {mean_abs[p]:.5f} "
          f"| {mean_abs[p]/mean_val:.1f}x global mean")


print("\nSaving individual URL heatmap")


ph_idx = np.where(y_sample == 1)[0]
domain_idx = [i for i in ph_idx if not is_ip_host(decode_seq(X_sample[i]))]

if domain_idx:
    best_idx = max(domain_idx, key=lambda i: y_prob_sample[i])
    print(f"    Selection: highest-confidence phishing URL with a domain hostname "
          f"({len(domain_idx)} of {len(ph_idx)} phishing URLs qualify)")
else:
    best_idx = int(ph_idx[np.argmax(y_prob_sample[ph_idx])]) if len(ph_idx) else 0
    print("    WARNING: no domain-hosted phishing URL in sample; using overall maximum")

case_url = decode_seq(X_sample[best_idx])
print(f"    Case study URL : {case_url}")

ind_sv = sv_2d[best_idx]
while ind_sv.ndim > 1:
    ind_sv = ind_sv.sum(axis=-1)
ind_sv = ind_sv.flatten()

top20      = np.argsort(np.abs(ind_sv))[-20:][::-1]
bar_colors = ["#C0392B" if v > 0 else "#27AE60" for v in ind_sv[top20]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.bar(range(20), ind_sv[top20], color=bar_colors, edgecolor="white")
ax1.set_xticks(range(20))
ax1.set_xticklabels([f"pos {i}" for i in top20], rotation=45, ha="right", fontsize=9)
ax1.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
ax1.set_ylabel("SHAP Value", fontsize=10)
ax1.set_title(f"Individual URL Top 20 Positions\n"
              f"Phishing Probability: {y_prob_sample[best_idx]*100:.1f}%",
              fontsize=12, fontweight="bold")
ax1.legend(handles=[
    mpatches.Patch(color="#C0392B", label="Toward PHISHING"),
    mpatches.Patch(color="#27AE60", label="Toward LEGITIMATE"),
], fontsize=9)
ax1.grid(axis="y", alpha=0.3)

# FIXED: use X_sample (the explained sample), not X_test
url_seq   = X_sample[best_idx].astype(int)
chars30   = [int_to_chr.get(int(c), "_") for c in url_seq[:30]]
shap30    = ind_sv[:30]
norm_shap = shap30 / (np.abs(shap30).max() + 1e-8)
cmap      = plt.cm.RdYlGn_r

for i, (ch, val) in enumerate(zip(chars30, norm_shap)):
    color = cmap((val + 1) / 2)
    ax2.add_patch(plt.Rectangle((i, 0), 1, 1, color=color, ec="white", lw=1))
    ax2.text(i + 0.5, 0.5, ch, ha="center", va="center", fontsize=10,
             fontweight="bold", color="white" if abs(val) > 0.5 else "black")

ax2.set_xlim(0, 30); ax2.set_ylim(0, 1)
ax2.set_xticks([]); ax2.set_yticks([])
ax2.set_title("Character Heatmap — First 30 Characters\n"
              "Red = pushes toward PHISHING  |  Green = pushes toward LEGITIMATE",
              fontsize=11, fontweight="bold")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
sm.set_array([])
plt.colorbar(sm, ax=ax2, orientation="horizontal", pad=0.1)
plt.tight_layout()
plt.savefig("shap_individual.png", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved: shap_individual.png")

# Case-study summary
top5      = np.argsort(np.abs(ind_sv))[-5:][::-1]
chars_all = [int_to_chr.get(int(c), "?") for c in url_seq]

print(f"\n  CASE STUDY SUMMARY (for Section 4.4.2)")
print(f"    Method     : shap.GradientExplainer")
print(f"    URL        : {case_url}")
print(f"    Verdict    : {'PHISHING' if y_prob_sample[best_idx] >= 0.5 else 'LEGITIMATE'}")
print(f"    True label : {'PHISHING' if y_sample[best_idx] == 1 else 'LEGITIMATE'}")
print(f"    Confidence : {y_prob_sample[best_idx]*100:.2f}%")
print(f"\n    Top 5 most influential character positions:")
for i in range(5):
    p   = int(top5[i])
    ch  = chars_all[p] if p < len(chars_all) else "?"
    val = ind_sv[p]
    print(f"      pos {p:3d} | char='{ch}' | SHAP={val:+.4f} | "
          f"{'-> PHISHING' if val > 0 else '-> LEGITIMATE'}")

print("\n  Output files:")
print("    shap_summary.png")
print("    shap_individual.png")
print("    shap_values.npy")