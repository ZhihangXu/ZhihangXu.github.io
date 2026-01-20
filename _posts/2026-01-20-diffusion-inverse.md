---
layout: post
title: "Notes on Diffusion Models as Plug-and-Play Priors"
date: 2026-01-20
author_profile: false
toc: true
read_time: true
tags: [diffusion, inverse-problems]
---

## Paper
Zheng et al. (2025); Tang et al. (2024); Huang et al. (2024).

## Problem
We consider high-dimensional inverse problems
$$
y = \mathcal{A}(x) + \varepsilon.
$$

## Key idea
Diffusion models represent the prior implicitly via the score
$$
\nabla_x \log p(x),
$$
which necessitates optimization or sampling for posterior inference.

## My notes
Although sampling-based approaches are asymptotically correct, probabilistic calibration in high-dimensional nonlinear settings remains unresolved.
