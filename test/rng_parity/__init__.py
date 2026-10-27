"""Independent CUDA reference assets for the CUDA-compatible RNG parity tests.

``cuda_philox_holdouts`` holds the 84 recorded H100 draws
(4 seeds x 7 sizes x {float32 normal, float32 exponential, float64 exponential})
with the CUDA generator state before and after each call; the product tree's
``rfantibody.rng`` must reproduce them from seed/offset/shape alone.
``cuda_mpnn_draws`` holds the two in-situ draws of the original deterministic
ProteinMPNN script (seed 42): the 478-element float32 normal used for the
decoding order and the first float64 ``multinomial`` event including its
captured probabilities, sampled index and exponential base variates.

These files are *expected outputs* only.  The product code never imports them.
"""
