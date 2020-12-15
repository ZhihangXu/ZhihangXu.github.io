---
layout:     post
title:      Marrying Up Regular Expressions with Neural Networks
subtitle:    "关于毕设的一篇ref paper"
date:       2020-12-15
author:     Chen Zhuo
header-img: img/2020-12-15.jpg
catalog: true
tags:
    - Paper
---


> 我的毕设参考的文章之一，也是一个很有趣的 idea。

# Marrying Up Regular Expressions with Neural Networks: A Case Study for Spoken Language Understanding

https://arxiv.org/abs/1805.05588

----

## Main idea / Motivation

​		应用RE融合NN，解决标注数据的**数量和质量**的限制。paper里这么总结：

> "Exploit the **conciseness and effectiveness** of REs and the **strong generalization ability** of NNs."

​		这篇工作主要在两个任务上进行实验：intent detection 和 slot filling。（其中intent detection 对我的帮助较大，这篇笔记更多注重intent detection。）

​		用RE encode的信息去 guide 神经网络的 attention mechanism。

## How to Evaluate

- ATIS 数据集

 - spoken language understanding (intent detection)与sentence classification相关
   - 输入句子，输出label
 - slot filling 与sequence labeling相关
   - 输入句子，输出label $y_i$'s of word $x_i$'s
	- （few shot情景）

## 在不同 level 融合RE和NN（主要工作）

1. **Base line 选取**：双向 LSTM，得到每个单词的hidden state $h_i$， 再把每个单词hidden state加权求和。

$$ s = \sum_i \alpha_i h_i \tag{1} $$, 

$\alpha_i$是这样算出来的：

$$\alpha_i = \frac{exp(h_i^TWc)}{\sum_i exp(h_i^TWc)} \tag{2}$$ 

$\alpha_i$ 也就是第i 个单词的 attention，c是随机初始化trainable的向量，用来选择有象征意义的单词，W是一个权重矩阵，维度是 (隐层维度 x c的维度。paper里也没写c具体是什么维度。最后$s$过一下softmax，分类。

2. **Input level:** 把RE给出的句子的 label （REtag）拼接到刚刚的 $s$上，或者把这个RE给出的label 拼接到每个单词的embedding上。他们抉择了一下，发现拼接到每个单词的方法会使网络严重依赖这个RE给出的tag，最后结果就和单枪匹马只用RE给句子分类效果相似了。所以最后选择把REtag只拼接到$s$上。

3. **Network module level**：简单的说就是上面（1）式，不再对任何一个句子都用同样的$\alpha$计算$s$，而是会用k个$\alpha$这里 k 等于句子的所有类别数。因为他们觉得对任意句子都用同样的attention计算句子embedding会使这个 attention less focused，然后相应的（1），（2）公式就会是：

$$ s_k = \sum_i \alpha_{ki} h_i \tag{1-2} $$ 

$$\alpha_{ki} = \frac{exp(h_i^TW_ac_k)}{\sum_i exp(h_i^TW_ac_k)} \tag{2-2}$$

$c_k$是一个针对类别k 的 trainable 的vector，$h_i$依旧是第i 个单词的双向LSTM的输出，$W_a$是一个weight matrix，也没说维度和含义。随后，判断一个句子属于第k 类是依照：

$$p_k = \frac{exp(logit_k)}{\sum_k exp(logit_k)} \tag{3}$$

其中$logit_k$：

$$logit_k = w_ks_k + b_k \tag{3-1}$$

但到这里还没完，他们说RE 如果给一个intent k 正值，说明这个句子属于这个intent，但是RE 也可以给出负值，代表这个句子不属于这个intent（==这个我还没太明白内部是什么原理，但应该不难，以后懂了再补上==），所以他们用正的RE logit减去负的RE logit 作为总的“把一个句子分为第 k 类的” logit。

$$logit_k=  logit_{pk} (positive \text{ } logit) - logit_{nk}(negative \text{ } logit) \tag{4}$$

然后为了让RE去引导attention，增加一个 attention loss：

$$loss_{attention \text{ } positive} = \sum_k \sum_i t_{ki} log(\alpha_{ki}) \tag{5}$$

$t_{ki}$是一个要么取0，要么取$\frac{1}{l_k}$的值，$l_k$是这个句子对应intent k 的clue words的个数，如果个数是0，则$t_{k*}$就=0。loss negative也是这样算(==我寻思着，这个loss不是只针对一个句子的loss吗，一个句子在所有k个intent上，最后attention loss算出来不是要么正要么负吗，那是不是也就是说这两个正负loss同时只有一个有意义，另一个就是0？==)。最终的loss:

$$loss = loss_c + \beta_ploss_{attention\_positive} + \beta loss_{attention \_ negative} \tag{6}$$

$loss_c$是原始分类的loss，$\beta$是两种attention loss的权重。（后面写到few shot时取16，其他设定下取1）

4. **Output level**：这个就很暴力了，直接用RE的预测 amend NN的预测：

$$logit_k = logit_k^{'}(RNN\_logit) + w_k z_k \tag{7}$$

$z-k$ 非0即1，零代表不曾有RE给这个句子分类为k，1代表至少有一个RE给这个句子分类为类别k。$w_k$是一个trainable weight，代表了RE的confidence，这里不给**各条**RE设置独立的trainable weight是因为一般只会有一些句子会被RE匹配。他们只修改logit而不修改最后的probability原因是，（这不是……差不多一个意思吗）logit是一个没有限制的实数，比概率更好的匹配了$w_k z_k$这一项的数学性质，他们也从实验上发现，修改logit比修改probability效果好。

<br/>
公式显示有点问题，后面解决一下，还有图床。
