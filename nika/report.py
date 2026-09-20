import time
from dataclasses import asdict

class TrainReport:
    def __init__(self, path, model, cfg, mcfg, dcfg):
        self.path = path
        self.model = model
        self.cfg, self.mcfg, self.dcfg = cfg, mcfg, dcfg
        self.rows = [] # (step, train, val, seconds)
        self.start = time.time()
        self.best_val = float("inf")
        self.best_step = None
        self.done = False
        self.write()

    def update(self, step, train_loss, val_loss):
        self.rows.append((step, train_loss, val_loss, time.time() - self.start))
        if val_loss < self.best_val:
            self.best_val, self.best_step = val_loss, step
        self.write()

    def finish(self):
        self.done = True
        self.write()

    # everything below just formats the file
    def write(self):
        with open(self.path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(self._lines()))

    def _lines(self):
        elapsed = time.time() - self.start
        status = "finished" if self.done else "running"
        n_params = sum(p.numel() for p in self.model.parameters())

        lines = [
            f"# Nika training report ({status})",
            "",
            f"- started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start))}",
            f"- elapsed: {elapsed / 60:.1f} min",
            f"- parameters: {n_params:,}",
            f"- steps: {self.rows[-1][0] if self.rows else 0} / {self.cfg.max_iters}",
            "",
            "## Config",
            "",
            "| setting | value |",
            "| --- | --- |",
        ]
        for name, c in (("model", self.mcfg), ("train", self.cfg), ("data", self.dcfg)):
            for k, v in asdict(c).items():
                lines.append(f"| {name}.{k} | {v} |")

        lines += ["", "## Model", "", "```", str(self.model), "```", ""]

        if self.rows:
            lines += [
                "## Losses",
                "",
                f"- best val: **{self.best_val:.3f}** at step {self.best_step}",
                f"- latest: train {self.rows[-1][1]:.3f} | val {self.rows[-1][2]:.3f}",
                "",
                "| step | train | val | gap | minutes |",
                "| --- | --- | --- | --- | --- |",
            ]
            for step, tr, va, secs in self.rows:
                lines.append(f"| {step} | {tr:.3f} | {va:.3f} | {va - tr:+.3f} | {secs / 60:.1f} |")
            lines += ["", "## Loss curve", "", "![loss](nika_loss.png)", ""]

        return lines
