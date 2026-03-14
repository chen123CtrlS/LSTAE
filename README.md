# LSTAE

This repository contains code for our paper "Anchoring Tracker: Long-Short Term Advantage Estimator".<img src="D:\桌面\RL_Reasoning\LSTAE\pics\overview.png" alt="overview" style="zoom:55%;" />

## 🎯 Introduction

Currently, mainstream RLVR methods predominantly adopt a "group-based" paradigm, with typical representatives including GRPO  and its variants (DAPO, GSPO). Although these approaches have achieved notable success by generating multiple responses per prompt and constructing on-policy advantage functions to reduce variance, training large models through RL in practice remains highly challenging:

- $\textcolor{purple}{\text{Sample efficiency is low}}$, such that more than one trajectory per prompt (G > 1) must be generated at each gradient step.

- In long-horizon reasoning tasks, $\textcolor{purple}{\text{sparse rewards often lead to high variance in advantage estimation}}$, thereby destabilizing the learning signal. For example, as long as the final result is correct, the model may still receive a positive reward even if the trajectory contains erroneous intermediate tool invocations—this inadvertently reinforces the model’s perception that such errors are acceptable behaviors.

- These methods $\textcolor{purple}{\text{fail to fully leverage the agent’s ability to accumulate experience}}$ from past trajectories and utilize it to guide future online optimization.

## 🎉key Insights
In this paper, we propose a novel single-stream algorithm—the Long-Short Term Advantage Estimator (LSTAE)—which adopts a two-dimensional credit assignment mechanism. $\textcolor{blue}{\text{The core idea is that the learning signal provided by each data point to the model is distinct and dynamically varying}}$.

LSTAE maintains a dedicated anchor historical experience tracker that leverages long-term (trajectory-level) and short-term (step-level) historical data to estimate multi-level advantages, thereby avoiding costly additional rollouts.

- **trajectory-level credit**: an adaptive anchor-based exploration reward mechanism facilitates reasoning diversity without compromising correctness.

- **step-level refinement**: the experience buffer of anchor states is leveraged to identify and group recurring states, thereby achieving fine-grained local credit assignment.


## ⚙️ Controlled Experiments

1. To install the experiment, please install the pip file.

```setup

```

>📋 You can adjust the following hyperparameters.
>

## 🚀 Realistic Data Verification

for math...

## 😊Other Results

