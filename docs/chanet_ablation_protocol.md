# CHANet whole-encoder ablation: paired B seed42

Status: implementation and local CPU checks complete; no formal B training started.
Baseline: b8c2eff. Reuse the completed A seed42 random-10-field run.

## Hypothesis and scope

Compare the full relation-aware graph encoder (A) with a relation-free node MLP
(B), keeping global pooling, raw node features, UAV state, action embedding,
hierarchical decoding, reward and PPO unchanged. This tests the whole graph
interaction/edge-relation encoding, not attention weights in isolation.
Handcrafted node features and global layout summaries remain in both arms.

B uses separate three-block residual MLPs for fields and boundaries. Every block
sees only its own node embedding and the same graph's UAV embedding. There are
no field-to-field, boundary-to-boundary or field-to-boundary messages, and no
edge attributes are read. UAV state is retained as broadcast conditioning rather
than relation-aware messages. Global pooling still allows the decoder to see all
nodes. Both single and disjoint-union batched paths use this encoder.

Default H=128, heads=4: A encoder 983,296 parameters; B 982,630 (difference
666, 0.068%). B hidden width 401 is determined by parameter count, not validation
performance. All parameters participate in computation; no dummy parameters.
The default/absent `rl.encoder.variant` remains `chanet`; B is `node_mlp`.

## Paired initialization and frozen controls

B starts with new optimizers and update zero. `--paired-init` must point to A's
`ckpt_update_0.pt`, not u1060 or any trained checkpoint. Copy identical-shaped
feature embedding, global encoder, action embedding, decoder and critic random
weights. B-only MLPs are randomly initialized from seed42. This is paired random
initialization, not fine-tuning. Record the reference file SHA256 and parameter
counts in `training_metadata.paired_initialization`.

The loader rejects nonzero source updates, nonrandom A initialization, wrong
encoder, metadata mismatch, or resolved-config differences other than variant.
Match seed, budget, world size, episodes/rank, instance seed base, max steps,
evaluation interval and both panel hashes. Fixed training config: 10 fields,
new instance each episode, 1,200 updates, 4 episodes/rank x4 ranks, actor LR 1e-4,
critic LR 3e-3, time reward, gamma1, same entropy schedule, eval every20.

## Workstation sequence

1. Sync this commit; preserve A and its complete output directory.
2. Run the targeted tests, then actual 4-GPU B smoke in a separate output:
   keep `--num-updates 1200 --eval-every 20`, add `--stop-at 3
   --checkpoint-every 1`. This stops the smoke without changing schedules.
3. Verify gate pass, no unused-parameter DDP failure, all ranks finite, parameter
   updates, variant/counts/shared-weight provenance, 12 generated instances/rank.
   Compare the first instance seeds/fingerprints to A's episode logs.
4. Check GPU checkpoint restore in a separate smoke output using `--resume` and
   `--stop-at 4`, without `--paired-init`; confirm no new initialization occurs.
5. If smoke passes, start formal B from A update0 in a NEW persistent directory;
   remove `--stop-at`, restore checkpoint-every10. Never resume smoke for formal B.
   Do not re-run A, change hyperparameters or run sealed tests automatically.

Example formal command (substitute existing workstation paths):

```bash
python -m coverage_env.rl.trainer.episode_random_worker \
  --diagnostic-bank /path/episode10/train10 \
  --validation-bank /path/episode10/valid30 \
  --output /path/episode10/ablation_node_mlp_s42 \
  --encoder-variant node_mlp \
  --paired-init /path/episode10/formal_random_instances_s42/ckpt_update_0.pt \
  --num-updates 1200 --episodes-per-rank 4 --seed 42 \
  --instance-seed-base 900000000 --actor-lr 1e-4 --critic-lr 3e-3 \
  --device cuda --checkpoint-every 10 --eval-every 20
```

If A update0 is missing or config checks fail, stop and report; do not substitute
trained weights or quietly alter A/B settings. Exact resume inherits the saved
encoder and rejects a conflicting variant. The original paired provenance stays
in metadata and does not require reloading A during resume.

## Analysis fixed before training

Compare validation-selected A u1060 to B's validation-selected checkpoint, and
A u1200 to B u1200. Report both, raw greedy curves, completion, total estimated
time, work/nonwork/maneuver/service, parameters and timings. Use B minus A costs
(positive means A is better). Diagnostic best never selects a model. These are
single paired-seed observations, not training-seed significance. Fresh-scale
panels already seen are exploratory; any later confirmation set must be frozen
before evaluating both models.

## Local validation

Three new integration tests cover edge sensitivity/invariance, parameter counts,
all-parameter gradients, single/batched outputs, shared update0 initialization,
rejection of trained checkpoints/config mismatch, CPU B training, save/restore
and continuous-versus-resumed equality. Existing encoder/rollout/resume tests
are also run. CUDA/NCCL are not available for this local review; workstation
GPU smoke is required before launching the authorized formal B run.
