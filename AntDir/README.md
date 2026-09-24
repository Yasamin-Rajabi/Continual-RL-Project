# AntDir (new benchmark; not the existing `ant/` velocity suite)

Suite: `ant_dir`. Eight fixed directions at 0,45,...,315 degrees, visited twice:
`0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7`. Task identity/direction is **not** appended
to the observation. Known sequence boundaries create a new task occurrence.
This fixed directional suite is a declared design choice, not a reproduction
of the PEARL randomly sampled training/test task split.

`antdir_envs.py` ports the reward and health termination in
`katerakelly/oyster/rlkit/envs/ant_dir.py` to Gymnasium's MuJoCo Ant-v4:

    reward = xy_torso_velocity dot [cos(direction), sin(direction)]
             + 1.0 - 0.5 * sum(action**2)
             - 0.0005 * sum(clip(contact_forces, -1, 1)**2)

Health termination: finite state and torso height in [0.2,1.0]. Time limit:
200 steps. Contact-force observations are included (expected observation
size 111, action size 8); verify this with the supplied real smoke test.

**Metrics:** Raw episodic return is the primary performance signal. To keep
the existing bounded-success metric pipeline usable, we additionally define a
custom per-step success diagnostic: healthy, projected speed >=0.2 m/s, and
heading error <=15 degrees. This is NOT a standard/published AntDir success
criterion. `velocity_error` is a compatibility alias for heading error in
radians; it is not a velocity error in m/s.

There is no finite reward upper bound supplied by this environment, so the
HalfCheetah `1-R/R_scratch` normalization must not be used. `FT_return` is NaN;
`FT_return_auc_delta` is the unnormalized continual-minus-scratch return AUC,
in raw return units. `FT_success` uses the explicitly custom success signal.

The new job files are starting presets: 300k interactions/occurrence, pool 5,
three seeds, deterministic evaluation, and Walker2D's remaining supplied job
settings. They have NOT been tuned or benchmarked. Existing HalfCheetah and
Walker2D presets were not changed. The default image is
`$HOME/containers/ethos_crl_torch280_mj237.sif`.

Use the repository-root `job_paper.sh --environments AntDir` launcher for the
new baselines and ablations. The local `job.sh` also supports the original
CKA/ETHOS workflow. See `../INTEGRATION_README.md`.

Reward source, retrieved for the port:
https://github.com/katerakelly/oyster/blob/master/rlkit/envs/ant_dir.py
Git blob: b0705ea65f99685ed92767933be1c08c9b5224b2
