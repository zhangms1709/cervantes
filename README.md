# cervantes

## Contents
- [Introduction](#introduction)
- [VanillaRNN](#vanillaRNN)
- [SeqGAN](#seqGAN)
- [Comparison](#comparison)
- [Literary musings](#literary-musings)

## Introduction

Many Text generation in Spanish (based on Don Quijote) using character-based RNNs

## Vanilla RNN

The first version of the RNN was designed to be a simple baseline model. It consists of one embedding layer, a long short-term memory (LSTM) layer, and a dense layer. 

\\begin{align*} f_t &= \\sigma \\left( W_f x_t + U_f h\_{t-1} + b_f
\\right) & \\textsf{Forget Module}\\ i_t &= \\sigma \\left( W_i x_t +
U_i h\_{t-1} + b_i \\right) & \\textsf{Remember Module}\\ \\tilde{c}*t
&= \\tanh \\left( W_c x_t + U_c h*{t-1} + b_c \\right) & \\textsf{New
Memory}\\ c_t &= f_t \\odot c\_{t-1} + i_t \\odot \\tilde{c}*t &
\\textsf{Cell State Update}\\ o_t &= \\sigma \\left( W_o x_t + U_o
h*{t-1} + b_o \\right) & \\textsf{Output Module}\\ h_t &= o_t \\odot
\\tanh(c_t) & \\textsf{Output, Hidden State Update}\\ \\end{align*}

![example model](model.png)

## SeqGAN

## Comparison

## Literary musings


