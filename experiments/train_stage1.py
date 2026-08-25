import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gradient_sharing_gate.data import dataset_labels, get_dataloader
from gradient_sharing_gate.gating import (
    apply_gate_to_update,
    moments_from_gradients,
    q_to_gate,
    sharing_q,
    update_moments,
)
from gradient_sharing_gate.seed import seed_everything
from gradient_sharing_gate.model import CLIP_MODEL, TwoStageCLIP, load_clip


def harmonic_mean(a, b):
    return 0.0 if a + b == 0 else 2.0 * a * b / (a + b)


def all_layernorm_parameters(method):
    method.model.requires_grad_(False)
    for encoder in (method.model.vision_model, method.model.text_model):
        for module in encoder.modules():
            if isinstance(module, nn.LayerNorm):
                module.requires_grad_(True)
    named = [(name, p) for name, p in method.model.named_parameters() if p.requires_grad]
    if not named:
        raise RuntimeError("No trainable vision/text LayerNorm parameters found")
    return named


def parameter_metadata(named):
    result, offset = [], 0
    for name, parameter in named:
        end = offset + parameter.numel()
        result.append({"name": name, "shape": list(parameter.shape), "start": offset, "end": end})
        offset = end
    return result


def flat_gradient(loss, named, retain_graph=False):
    grads = torch.autograd.grad(
        loss,
        [parameter for _, parameter in named],
        retain_graph=retain_graph,
        allow_unused=True,
    )
    pieces = [
        torch.zeros_like(parameter).reshape(-1) if grad is None else grad.reshape(-1)
        for grad, (_, parameter) in zip(grads, named)
    ]
    return torch.cat(pieces)


class OutcomeBalancedCycleStream:
    def __init__(self, labels, initial_outcomes, seed):
        self.rng = random.Random(seed)
        self.pairs = [(index, int(label)) for index, label in enumerate(labels)]
        self.outcomes = dict(initial_outcomes)
        self.current_results = {}
        self.queue = []

    def _start_cycle(self):
        if self.current_results:
            if len(self.current_results) != len(self.pairs):
                raise RuntimeError("validation cycle ended before every image was observed")
            self.outcomes = self.current_results
            self.current_results = {}
        successes = [pair for pair in self.pairs if self.outcomes[pair]]
        failures = [pair for pair in self.pairs if not self.outcomes[pair]]
        self.rng.shuffle(successes)
        self.rng.shuffle(failures)
        queue, si, fi = [], 0, 0
        for prefix_size in range(1, len(self.pairs) + 1):
            target = round(prefix_size * len(successes) / len(self.pairs))
            if si < target:
                queue.append(successes[si])
                si += 1
            else:
                queue.append(failures[fi])
                fi += 1
        self.queue = queue

    def next(self, count):
        if not self.queue:
            self._start_cycle()
        take = min(count, len(self.queue))
        result = self.queue[:take]
        del self.queue[:take]
        return result

    def record(self, pairs, correctness):
        for pair, correct in zip(pairs, correctness):
            if pair in self.current_results:
                raise RuntimeError("validation image repeated within a cycle")
            self.current_results[pair] = bool(correct)

    def state_dict(self):
        return {
            "rng_state": self.rng.getstate(),
            "outcomes": self.outcomes,
            "current_results": self.current_results,
            "queue": self.queue,
        }

    def load_state_dict(self, state):
        self.rng.setstate(state["rng_state"])
        self.outcomes = state["outcomes"]
        self.current_results = state["current_results"]
        self.queue = state["queue"]


def images_and_labels(dataset, pairs, device):
    images = [dataset[index][0] for index, _ in pairs]
    labels = [label for _, label in pairs]
    return torch.stack(images).to(device), torch.tensor(labels, device=device)


@torch.no_grad()
def classifiers(method, new_classes):
    return method.encode_text(), method.encode_classnames(new_classes)


@torch.no_grad()
def full_accuracy_and_outcomes(method, dataset, classifier, device, batch_size=128):
    labels = [int(label) for label in dataset_labels(dataset)]
    pairs = [(index, label) for index, label in enumerate(labels)]
    outcomes = {}
    method.eval()
    for start in range(0, len(pairs), batch_size):
        part = pairs[start:start + batch_size]
        images, targets = images_and_labels(dataset, part, device)
        predictions = method.classifier_logits(images, classifier).argmax(-1)
        outcomes.update(zip(part, (predictions == targets).tolist()))
    return sum(outcomes.values()) / len(outcomes), outcomes


@torch.no_grad()
def sampled_accuracy(method, dataset, pairs, classifier, device):
    images, targets = images_and_labels(dataset, pairs, device)
    predictions = method.classifier_logits(images, classifier).argmax(-1)
    correctness = predictions == targets
    return correctness.float().mean().item(), correctness.tolist()


def initialize_q(method, train_dataset, named, count, device):
    python_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    generator = torch.Generator().manual_seed(9137)
    indices = torch.randperm(len(train_dataset), generator=generator)[: min(count, len(train_dataset))].tolist()
    gradients = []
    method.train()
    for start in range(0, len(indices), 16):
        part = indices[start:start + 16]
        batch = [train_dataset[index] for index in part]
        images = torch.stack([item[0] for item in batch]).to(device)
        labels = torch.tensor([int(item[1]) for item in batch], device=device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            losses = F.cross_entropy(method.stage_one_logits(images), labels, reduction="none")
        for index, loss in enumerate(losses):
            gradients.append(flat_gradient(loss, named, retain_graph=index + 1 < len(losses)).detach())
    first_abs, second_square = moments_from_gradients(gradients)
    random.setstate(python_state)
    torch.random.set_rng_state(torch_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
    return first_abs, second_square


class QVectorWriter:
    def __init__(self, directory, chunk_steps, metadata):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.chunk_steps = chunk_steps
        self.steps, self.values = [], []
        (self.directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def add(self, step, q):
        self.steps.append(step)
        self.values.append(q.detach().to(dtype=torch.float16, device="cpu"))
        if len(self.steps) >= self.chunk_steps:
            self.flush()

    def flush(self):
        if not self.steps:
            return
        start, end = self.steps[0], self.steps[-1]
        destination = self.directory / f"q_steps_{start:04d}_{end:04d}.pt"
        if destination.exists():
            raise FileExistsError(destination)
        torch.save({"steps": torch.tensor(self.steps), "q": torch.stack(self.values)}, destination)
        self.steps, self.values = [], []


def save_checkpoint(
    destination, step, named, optimizer, scheduler, scaler, first_abs,
    second_square, ema_base, ema_new, initial_base, initial_new,
    base_stream, new_stream, args,
):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    state = {
        "version": 1,
        "step": step,
        "method": args.method,
        "dataset": args.dataset,
        "seed": args.seed,
        "steps": args.steps,
        "parameter_metadata": parameter_metadata(named),
        "trainable_parameters": {
            name: parameter.detach().cpu() for name, parameter in named
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "first_abs": None if first_abs is None else first_abs.detach().cpu(),
        "second_square": None if second_square is None else second_square.detach().cpu(),
        "ema_base": ema_base,
        "ema_new": ema_new,
        "initial_base": initial_base,
        "initial_new": initial_new,
        "base_stream": base_stream.state_dict(),
        "new_stream": new_stream.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, temporary)
    temporary.replace(destination)


def restore_checkpoint(
    checkpoint, named, optimizer, scheduler, scaler, device,
    restore_scheduler=True,
):
    if checkpoint["parameter_metadata"] != parameter_metadata(named):
        raise RuntimeError("checkpoint LayerNorm parameters do not match the current model")
    for name, parameter in named:
        parameter.data.copy_(checkpoint["trainable_parameters"][name].to(device))
    optimizer.load_state_dict(checkpoint["optimizer"])
    if restore_scheduler:
        scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])


def run(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loaders, _, _ = get_dataloader(
        args.batch_size, args.dataset, root=args.data_root,
        shots=args.shots, setting="base2new",
    )
    val_base_loader, val_new_loader = validation_loaders
    model, tokenizer = load_clip(CLIP_MODEL)
    method = TwoStageCLIP(
        model, tokenizer, train_loader.dataset.classes, train_loader.dataset.template
    ).to(device)
    named = all_layernorm_parameters(method)
    optimizer = torch.optim.AdamW([p for _, p in named], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.steps, eta_min=args.lr_min
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    use_q = args.method == "q"
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    checkpoint = None
    if args.resume and checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["method"] != args.method or checkpoint["seed"] != args.seed:
            raise RuntimeError("checkpoint method/seed does not match this run")
        extending = checkpoint["steps"] < args.steps and args.extend
        if checkpoint["steps"] != args.steps and not extending:
            raise RuntimeError("checkpoint total steps does not match --steps; use --extend to continue farther")
        restore_checkpoint(
            checkpoint, named, optimizer, scheduler, scaler, device,
            restore_scheduler=not extending,
        )
        if extending:
            for group in optimizer.param_groups:
                group["lr"] = args.extension_lr
                group["initial_lr"] = args.extension_lr
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, args.steps - checkpoint["step"], eta_min=args.lr_min
            )
            print(
                f"extending from step {checkpoint['step']} to {args.steps} "
                f"with lr={args.extension_lr:g}->{args.lr_min:g}",
                flush=True,
            )

    first_abs = second_square = None
    q_writer = None
    if use_q:
        if checkpoint is None:
            first_abs, second_square = initialize_q(
                method, train_loader.dataset, named, args.q_init_images, device
            )
        else:
            first_abs = checkpoint["first_abs"].to(device)
            second_square = checkpoint["second_square"].to(device)
        q_writer = QVectorWriter(
            args.q_output,
            args.q_chunk_steps,
            {
                "version": 1,
                "dataset": args.dataset,
                "method": args.method,
                "seed": args.seed,
                "steps": args.steps,
                "q_dtype": "float16",
                "q_definition": "E_abs_g_squared_over_E_g_squared",
                "q_ema_beta": args.q_ema_beta,
                "q_gate_max": args.q_gate_max,
                "parameters": parameter_metadata(named),
            },
        )

    method.eval()
    base_classifier, new_classifier = classifiers(method, val_new_loader.dataset.classes)
    if checkpoint is None:
        initial_base, base_outcomes = full_accuracy_and_outcomes(
            method, val_base_loader.dataset, base_classifier, device
        )
        initial_new, new_outcomes = full_accuracy_and_outcomes(
            method, val_new_loader.dataset, new_classifier, device
        )
        base_stream = OutcomeBalancedCycleStream(
            dataset_labels(val_base_loader.dataset), base_outcomes, args.seed + 101
        )
        new_stream = OutcomeBalancedCycleStream(
            dataset_labels(val_new_loader.dataset), new_outcomes, args.seed + 202
        )
        ema_base, ema_new = initial_base, initial_new
        start_step = 1
    else:
        initial_base = checkpoint["initial_base"]
        initial_new = checkpoint["initial_new"]
        base_stream = OutcomeBalancedCycleStream(
            dataset_labels(val_base_loader.dataset), checkpoint["base_stream"]["outcomes"], args.seed + 101
        )
        new_stream = OutcomeBalancedCycleStream(
            dataset_labels(val_new_loader.dataset), checkpoint["new_stream"]["outcomes"], args.seed + 202
        )
        base_stream.load_state_dict(checkpoint["base_stream"])
        new_stream.load_state_dict(checkpoint["new_stream"])
        ema_base, ema_new = checkpoint["ema_base"], checkpoint["ema_new"]
        start_step = checkpoint["step"] + 1
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.random.set_rng_state(checkpoint["torch_rng"])
        if checkpoint["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        print(f"resumed {args.method} seed={args.seed} from step {checkpoint['step']}", flush=True)

    output = Path(args.metrics_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method", "seed", "step", "loss", "query_base", "query_new",
        "query_hm", "ema_base", "ema_new", "ema_hm", "q_mean", "q_std",
        "q_min", "q_max", "gate_mean", "seconds_per_step", "initial_full_base",
        "initial_full_new", "initial_full_hm", "final_full_base", "final_full_new",
        "final_full_hm", "full_base", "full_new", "full_hm",
    ]
    append = checkpoint is not None
    if append:
        if not output.exists():
            raise FileNotFoundError(f"metrics CSV missing for checkpoint: {output}")
        with output.open("r", newline="", encoding="utf-8") as existing:
            rows = list(csv.DictReader(existing))
        rows = [row for row in rows if int(row["step"]) <= checkpoint["step"]]
        if not rows or int(rows[-1]["step"]) != checkpoint["step"]:
            raise RuntimeError("metrics CSV does not contain the checkpoint step")
        with output.open("w", newline="", encoding="utf-8") as repaired:
            repair_writer = csv.DictWriter(repaired, fieldnames=fields)
            repair_writer.writeheader()
            repair_writer.writerows(rows)
    with output.open("a" if append else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not append:
            writer.writeheader()
        iterator = iter(train_loader)
        started = time.perf_counter()
        for step in range(start_step, args.steps + 1):
            try:
                images, labels = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                images, labels = next(iterator)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                losses = F.cross_entropy(method.stage_one_logits(images), labels, reduction="none")

            q = gate = None
            previous = None
            if use_q:
                selected = [((step - 1) * args.q_online_images + i) % len(losses) for i in range(args.q_online_images)]
                observed = [
                    flat_gradient(losses[index], named, retain_graph=True).detach()
                    for index in selected
                ]
                first_abs, second_square = update_moments(
                    first_abs, second_square, observed, args.q_ema_beta
                )
                q = sharing_q(first_abs, second_square, args.epsilon)
                gate = q_to_gate(q, args.q_gate_max)
                q_writer.add(step, q)
                previous = [parameter.detach().clone() for _, parameter in named]

            scale_before = scaler.get_scale()
            scaler.scale(losses.mean()).backward()
            scaler.step(optimizer)
            scaler.update()
            step_succeeded = scaler.get_scale() >= scale_before
            if use_q and step_succeeded:
                apply_gate_to_update(named, previous, gate)
            if step_succeeded:
                scheduler.step()

            method.eval()
            with torch.no_grad():
                base_classifier, new_classifier = classifiers(method, val_new_loader.dataset.classes)
            base_pairs = base_stream.next(args.validation_images_per_group)
            new_pairs = new_stream.next(args.validation_images_per_group)
            query_base, base_correctness = sampled_accuracy(
                method, val_base_loader.dataset, base_pairs, base_classifier, device
            )
            query_new, new_correctness = sampled_accuracy(
                method, val_new_loader.dataset, new_pairs, new_classifier, device
            )
            base_stream.record(base_pairs, base_correctness)
            new_stream.record(new_pairs, new_correctness)
            ema_base = args.validation_ema_beta * ema_base + (1.0 - args.validation_ema_beta) * query_base
            ema_new = args.validation_ema_beta * ema_new + (1.0 - args.validation_ema_beta) * query_new

            full_base = full_new = full_hm = ""
            should_run_full_validation = step == args.steps or (
                args.full_validation_every > 0
                and step % args.full_validation_every == 0
            )
            if should_run_full_validation:
                full_base, _ = full_accuracy_and_outcomes(
                    method, val_base_loader.dataset, base_classifier, device
                )
                full_new, _ = full_accuracy_and_outcomes(
                    method, val_new_loader.dataset, new_classifier, device
                )
                full_hm = harmonic_mean(full_base, full_new)
            final_base = full_base if step == args.steps else ""
            final_new = full_new if step == args.steps else ""
            final_hm = full_hm if step == args.steps else ""
            row = {
                "method": args.method, "seed": args.seed, "step": step,
                "loss": losses.mean().item(), "query_base": query_base,
                "query_new": query_new, "query_hm": harmonic_mean(query_base, query_new),
                "ema_base": ema_base, "ema_new": ema_new,
                "ema_hm": harmonic_mean(ema_base, ema_new),
                "q_mean": "" if q is None else q.mean().item(),
                "q_std": "" if q is None else q.std(unbiased=False).item(),
                "q_min": "" if q is None else q.min().item(),
                "q_max": "" if q is None else q.max().item(),
                "gate_mean": "" if gate is None else gate.mean().item(),
                "seconds_per_step": (time.perf_counter() - started) / (step - start_step + 1),
                "initial_full_base": initial_base, "initial_full_new": initial_new,
                "initial_full_hm": harmonic_mean(initial_base, initial_new),
                "final_full_base": final_base, "final_full_new": final_new,
                "final_full_hm": final_hm,
                "full_base": full_base, "full_new": full_new,
                "full_hm": full_hm,
            }
            writer.writerow(row)
            handle.flush()
            should_checkpoint = checkpoint_path is not None and (
                step == args.steps or (
                    args.checkpoint_every > 0 and step % args.checkpoint_every == 0
                )
            )
            if should_checkpoint:
                if q_writer is not None:
                    q_writer.flush()
                save_checkpoint(
                    checkpoint_path, step, named, optimizer, scheduler, scaler,
                    first_abs, second_square, ema_base, ema_new, initial_base,
                    initial_new, base_stream, new_stream, args,
                )
                print(f"checkpoint -> {checkpoint_path} (step {step})", flush=True)
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                print(
                    f"{args.method} seed={args.seed} [{step}/{args.steps}] "
                    f"loss={row['loss']:.4f} ema=({100*ema_base:.2f},"
                    f"{100*ema_new:.2f},{100*row['ema_hm']:.2f}) "
                    f"sec/step={row['seconds_per_step']:.2f}", flush=True
                )
            method.train()
    if q_writer is not None:
        q_writer.flush()
    print(f"metrics -> {output}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="2SFS Stage 1 raw/q paired-seed experiment")
    parser.add_argument("--method", choices=["raw", "q"], required=True)
    parser.add_argument("--dataset", default="fgvc")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--validation_images_per_group", type=int, default=16)
    parser.add_argument("--validation_ema_beta", type=float, default=0.97)
    parser.add_argument("--full_validation_every", type=int, default=0)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extend", action="store_true")
    parser.add_argument("--extension_lr", type=float, default=2e-5)
    parser.add_argument("--q_init_images", type=int, default=200)
    parser.add_argument("--q_online_images", type=int, default=4)
    parser.add_argument("--q_ema_beta", type=float, default=0.95)
    parser.add_argument("--q_gate_max", type=float, default=0.5)
    parser.add_argument("--q_chunk_steps", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=1e-30)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--metrics_output", required=True)
    parser.add_argument("--q_output", default="gradient_gate_data/q_vectors")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
