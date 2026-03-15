"""
Full SpiNNaker Deployment: IF_cond_exp (conductance-based) + MaxPool model.

Attempts full FC1+FC2 deployment on SpiNNaker using conductance-based
synapses (IF_cond_exp) instead of current-based (IF_curr_exp).

Key insight: conductance-based models prevent excitatory-inhibitory
cancellation because inhibition is SHUNTING (g_syn * (V - E_rev)),
not subtractive. The membrane cannot go below E_rev_I, preventing
the catastrophic cancellation that killed the original FC1 deployment.

Prerequisites:
    1. Train MaxPool SNN model (Option A) — use spinnaker_option_a.py
    2. Run with .venv-spinnaker environment
    3. Access to SpiNNaker via spalloc

Usage (requires .venv-spinnaker):
    python experiments/full_spinnaker_deploy_cond.py --fold 4 --num-samples 20
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DIR, NUM_STEPS


def load_model_weights(fold=4, encoding="direct"):
    """Load FC1 and FC2 weights from trained snnTorch model."""
    import torch
    from src.models.snn_model import SpikingCNN

    model = SpikingCNN()
    model_path = RESULTS_DIR / "snn" / encoding / f"best_fold{fold}.pt"
    model.load_state_dict(torch.load(model_path, weights_only=True))

    fc1_w = model.fc1.weight.detach().numpy()  # (256, 2304)
    fc1_b = model.fc1.bias.detach().numpy() if model.fc1.bias is not None else None
    fc2_w = model.fc2.weight.detach().numpy()  # (50, 256)
    fc2_b = model.fc2.bias.detach().numpy() if model.fc2.bias is not None else None

    return {
        "fc1_weight": fc1_w, "fc1_bias": fc1_b,
        "fc2_weight": fc2_w, "fc2_bias": fc2_b,
    }


def prepare_connection_lists(weight_matrix, weight_scale=1.0, prune_threshold=0.01):
    """Convert weight matrix to sPyNNaker FromListConnector format.

    For IF_cond_exp, weights must be positive (conductances in uS).
    Excitatory and inhibitory are separated by receptor_type.

    Returns:
        exc_conns: list of (pre, post, weight, delay) for excitatory
        inh_conns: list of (pre, post, weight, delay) for inhibitory
    """
    exc_conns = []
    inh_conns = []
    n_post, n_pre = weight_matrix.shape

    for post in range(n_post):
        for pre in range(n_pre):
            w = weight_matrix[post, pre] * weight_scale
            if abs(w) < prune_threshold:
                continue
            if w > 0:
                exc_conns.append((pre, post, float(w), 1.0))
            else:
                inh_conns.append((pre, post, float(abs(w)), 1.0))

    return exc_conns, inh_conns


def extract_input_features(fold=4, encoding="direct", num_samples=20):
    """Extract features after conv+pool+avgpool for FC1 input.

    Uses snnTorch to run the conv layers and extract the flattened
    features that FC1 would receive.
    """
    import torch
    from src.config import BATCH_SIZE, get_device
    from src.dataset import get_fold_dataloaders
    from src.encoding import get_encoder
    from src.models.snn_model import SpikingCNN

    device = get_device()
    _, test_loader = get_fold_dataloaders(fold, batch_size=1)
    encoder = get_encoder(encoding)

    model = SpikingCNN().to(device)
    model_path = RESULTS_DIR / "snn" / encoding / f"best_fold{fold}.pt"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    all_features = []  # (N, 25, 2304)
    all_labels = []
    all_snntorch_preds = []

    with torch.no_grad():
        for i, (data, targets) in enumerate(test_loader):
            if i >= num_samples:
                break

            data = data.to(device)
            spk_input = encoder(data).to(device)

            # Run conv layers, extract flattened features
            mem1 = model.lif1.init_leaky()
            mem2 = model.lif2.init_leaky()
            mem3 = model.lif3.init_leaky()
            mem4 = model.lif4.init_leaky()

            features_per_step = []
            mem_out_acc = torch.zeros(1, 50, device=device)

            for step in range(NUM_STEPS):
                x_t = spk_input[step]
                cur1 = model.pool1(model.bn1(model.conv1(x_t)))
                spk1, mem1 = model.lif1(cur1, mem1)
                cur2 = model.pool2(model.bn2(model.conv2(spk1)))
                spk2, mem2 = model.lif2(cur2, mem2)
                pooled = model.avg_pool(spk2)
                flat = pooled.view(pooled.size(0), -1)  # (1, 2304)
                features_per_step.append(flat.cpu().numpy()[0])

                # Also get full model prediction for comparison
                cur3 = model.fc1(flat)
                spk3, mem3 = model.lif3(cur3, mem3)
                cur4 = model.fc2(spk3)
                spk4, mem4 = model.lif4(cur4, mem4)
                mem_out_acc += mem4

            all_features.append(np.array(features_per_step))  # (25, 2304)
            all_labels.append(targets[0].item())
            all_snntorch_preds.append(mem_out_acc.argmax(dim=1).item())

    return (
        np.array(all_features),  # (N, 25, 2304)
        np.array(all_labels),
        np.array(all_snntorch_preds),
    )


def run_spinnaker_full(features, labels, weights, weight_scale=1.0,
                       prune_threshold=0.01):
    """Deploy FC1+FC2 on SpiNNaker using IF_cond_exp."""
    try:
        import pyNN.spiNNaker as sim
    except ImportError:
        print("ERROR: sPyNNaker not available. Use .venv-spinnaker.")
        return None

    n_samples = features.shape[0]
    fc1_exc, fc1_inh = prepare_connection_lists(
        weights["fc1_weight"], weight_scale, prune_threshold
    )
    fc2_exc, fc2_inh = prepare_connection_lists(
        weights["fc2_weight"], weight_scale, prune_threshold
    )

    print(f"  FC1: {len(fc1_exc)} exc + {len(fc1_inh)} inh connections")
    print(f"  FC2: {len(fc2_exc)} exc + {len(fc2_inh)} inh connections")

    # Conductance-based LIF parameters
    cond_params_hidden = {
        "cm": 1.0,
        "tau_m": 20.0,
        "tau_refrac": 0.1,
        "v_reset": -65.0,
        "v_rest": -65.0,
        "v_thresh": -50.0,
        "tau_syn_E": 5.0,
        "tau_syn_I": 5.0,
        "e_rev_E": 0.0,
        "e_rev_I": -80.0,
    }

    cond_params_output = {
        "cm": 1.0,
        "tau_m": 20.0,
        "tau_refrac": 0.1,
        "v_reset": -65.0,
        "v_rest": -65.0,
        "v_thresh": -50.0,
        "tau_syn_E": 5.0,
        "tau_syn_I": 5.0,
        "e_rev_E": 0.0,
        "e_rev_I": -80.0,
    }

    results = []

    for idx in range(n_samples):
        sample_features = features[idx]  # (25, 2304)
        true_label = int(labels[idx])

        # Convert continuous features to spike times
        # Threshold the features to get binary spikes
        threshold = 0.1  # Tune this
        spike_times = {}
        for j in range(2304):
            times = []
            for t in range(NUM_STEPS):
                if sample_features[t, j] > threshold:
                    times.append(float(t + 1))
            spike_times[j] = times

        try:
            sim.setup(timestep=1.0)

            # Input population (2304 neurons = flattened conv features)
            input_pop = sim.Population(
                2304,
                sim.SpikeSourceArray(
                    spike_times=[spike_times[j] for j in range(2304)]
                ),
                label="input"
            )

            # Hidden population (256 neurons, IF_cond_exp)
            hidden_pop = sim.Population(
                256,
                sim.IF_cond_exp(**cond_params_hidden),
                label="hidden"
            )

            # Output population (50 neurons, IF_cond_exp)
            output_pop = sim.Population(
                50,
                sim.IF_cond_exp(**cond_params_output),
                label="output"
            )

            # Record membrane voltages for classification
            output_pop.record(["v"])
            hidden_pop.record(["spikes"])

            # FC1 projections
            if fc1_exc:
                sim.Projection(input_pop, hidden_pop,
                              sim.FromListConnector(fc1_exc),
                              receptor_type="excitatory")
            if fc1_inh:
                sim.Projection(input_pop, hidden_pop,
                              sim.FromListConnector(fc1_inh),
                              receptor_type="inhibitory")

            # FC2 projections
            if fc2_exc:
                sim.Projection(hidden_pop, output_pop,
                              sim.FromListConnector(fc2_exc),
                              receptor_type="excitatory")
            if fc2_inh:
                sim.Projection(hidden_pop, output_pop,
                              sim.FromListConnector(fc2_inh),
                              receptor_type="inhibitory")

            sim.run(NUM_STEPS)

            # Read results
            v_data = output_pop.get_data("v")
            v_signal = v_data.segments[0].analogsignals[0]
            final_v = np.array([v_signal.magnitude[:, n].sum() for n in range(50)])
            predicted = int(np.argmax(final_v))

            # Count hidden spikes
            hidden_spikes = hidden_pop.get_data("spikes")
            n_hidden_spikes = len(hidden_spikes.segments[0].spiketrains[0]) if hidden_spikes.segments[0].spiketrains else 0
            total_hidden = sum(len(st) for st in hidden_spikes.segments[0].spiketrains)

            sim.end()

            correct = int(predicted == true_label)
            results.append({
                "sample": idx,
                "true": true_label,
                "predicted": predicted,
                "correct": correct,
                "hidden_spikes_total": int(total_hidden),
            })

            status = "CORRECT" if correct else "WRONG"
            print(f"  [{idx+1}/{n_samples}] True={true_label} Pred={predicted} "
                  f"{status} | Hidden spikes: {total_hidden}")

        except Exception as e:
            sim.end()
            print(f"  [{idx+1}/{n_samples}] ERROR: {e}")
            results.append({
                "sample": idx, "true": true_label,
                "predicted": -1, "correct": 0, "error": str(e),
            })

    accuracy = sum(r["correct"] for r in results) / len(results) if results else 0
    return results, accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Full SpiNNaker deploy with IF_cond_exp"
    )
    parser.add_argument("--fold", type=int, default=4)
    parser.add_argument("--encoding", default="direct")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--weight-scale", type=float, default=0.01,
                        help="Weight scale for conductance values (default: 0.01)")
    parser.add_argument("--prune-threshold", type=float, default=0.01)
    parser.add_argument("--extract-only", action="store_true",
                        help="Only extract features, don't run SpiNNaker")
    args = parser.parse_args()

    save_dir = RESULTS_DIR / "spinnaker_results" / "full_deploy_cond"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Full SpiNNaker Deployment (IF_cond_exp)")
    print("=" * 60)

    # Step 1: Load weights
    print("\n[1/3] Loading model weights...")
    weights = load_model_weights(args.fold, args.encoding)
    print(f"  FC1: {weights['fc1_weight'].shape}")
    print(f"  FC2: {weights['fc2_weight'].shape}")

    # Step 2: Extract input features
    print("\n[2/3] Extracting conv features via snnTorch...")
    features, labels, snntorch_preds = extract_input_features(
        args.fold, args.encoding, args.num_samples
    )
    print(f"  Features shape: {features.shape}")
    print(f"  Feature stats: min={features.min():.4f} max={features.max():.4f} "
          f"mean={features.mean():.4f}")

    # Save features
    np.save(save_dir / "fc1_input_features.npy", features)
    np.save(save_dir / "labels.npy", labels)

    snntorch_acc = (snntorch_preds == labels).mean()
    print(f"  snnTorch full model accuracy: {snntorch_acc:.4f}")

    if args.extract_only:
        print("\n--extract-only: stopping before SpiNNaker.")
        return

    # Step 3: SpiNNaker deployment
    print(f"\n[3/3] Running on SpiNNaker (weight_scale={args.weight_scale})...")
    results, accuracy = run_spinnaker_full(
        features, labels, weights,
        weight_scale=args.weight_scale,
        prune_threshold=args.prune_threshold,
    )

    if results:
        summary = {
            "fold": args.fold,
            "encoding": args.encoding,
            "num_samples": len(results),
            "weight_scale": args.weight_scale,
            "prune_threshold": args.prune_threshold,
            "neuron_model": "IF_cond_exp",
            "spinnaker_accuracy": accuracy,
            "snntorch_accuracy": float(snntorch_acc),
            "hardware_gap": float(snntorch_acc - accuracy),
            "results": results,
        }

        with open(save_dir / f"full_deploy_fold{args.fold}.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Full Deploy Results (Fold {args.fold})")
        print(f"{'='*60}")
        print(f"  SpiNNaker (IF_cond_exp): {accuracy:.4f}")
        print(f"  snnTorch:               {snntorch_acc:.4f}")
        print(f"  Hardware gap:           {snntorch_acc - accuracy:.4f}")
        print(f"\nSaved: {save_dir / f'full_deploy_fold{args.fold}.json'}")


if __name__ == "__main__":
    main()
