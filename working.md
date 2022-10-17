# 概要

- motivation:
  - M1: prompt 鲁棒性 (待进一步验证，和全量微调比)
  - M2: prompt 微调速度
- preliminaries
  - 对于DNN, 浅层网络学习基础特征，深层网络学习目标相关特征 **([1]调研：相关论文)**
  - 假设：只对深层网络调整，保留浅层网络提取特征的能力，对鲁棒性有好处
- solution
  - for M1: 只在深层添加 soft prompt
  - for M2: cache feature **([1]调研：类似做法的论文)**



# 相关论文



# prompt

## survey

https://github.com/thunlp/PromptPapers

 

## promtBERT

> paper: PromptBERT: Improving BERT Sentence Embeddings with Prompts
>
> author: Ting Jiang1, Shaohan Huang, Zihan Zhang, Deqing Wang, Fuzhen Zhuang , Furu Wei, Haizhen Huang, Liangjie Zhang , Qi Zhang
>
> institute: Beihang University, Microsoft
>
> main: 这篇文章观察发现了 sentence embedding 中的bias，包括frequency bias, subword and case bias. 提出的解决方案是用promt 的方法，来或取sentence embedding，如table4 所示, X 是 sentence，根据[MASK] 获取sentence embedding. 共分为两步：1. 获取模板，分成手工设计和搜索两种方式。2. Represent Sentence with the Prompt，作者提出了两种方法 a. 用 MASK 对应的embedding b. get top-k tokens according to h[MASK] and MLM classification head, 然后加权平均
>
> ![image-20220718204948639](.\working.assets\image-20220718204948639.png)
>
> related
>
> - prompt search：template generation based T5 (Gao et al., 2020) and OptiPrompt (Zhong et al., 2021).

![image-20220718212927461](.\working.assets\image-20220718212927461.png)



## prefix tuning

> paper: [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://aclanthology.org/2021.acl-long.353.pdf) (ACL 2021)
>
> author: Xiang Lisa Li， Percy Liang
>
> institute: stanford
>
> code：
>
> - Our code is available at https://github.com/XiangLi1999/PrefixTuning. 
>
> - Experiments and data are available at https://worksheets.codalab.org/worksheets/0x16e0c8e7ab1f4b22aaccddc8b586541f
>
> main：跟p-tuning v2 区别不大，场景更多一些，包括NLG, encoder-decoder model；有明显区别的地方：
>
> - 有一个 parameterization of $P_{\theta}$ 的操作，也就是对prefix embedding 经过一层MLP 的变换操作，相当于p-tuning v2 代码中开启projection 功能
>
> points：
>
> - 研究了用不同真实词来初始化prefix
>
>   ![image-20220718214747946](.\working.assets\image-20220718214747946.png)
>
> - prefix 长度
>
>   ![image-20220718214914950](.\working.assets\image-20220718214914950.png)
>
> - 数据量
>
>   ![image-20220718214858935](.\working.assets\image-20220718214858935.png)



## The Power of Scale for Parameter-Efficient Prompt Tuning

> paper: [The Power of Scale for Parameter-Efficient Prompt Tuning](https://aclanthology.org/2021.emnlp-main.243.pdf) (EMNLP 2021)
>
> author: Brian Lester，Rami Al-Rfou Noah Constant
>
> institute：Google
>
> code: https://github.com/google-research/prompt-tuning
>
> main: 这偏文章跟prefix tuning很像，但是只在第一层加了 promt token， 而prefix tuning 在每层都加了。主要内容包括：讨论了prefix length 和 initialization 的影响，prefix tuning on domin shift， prompt ensembling。
>
> ![image-20220718221905461](.\working.assets\image-20220718221905461.png)
>
> points：
>
> - initailization designs:
>
>   - random initialization
>   - initialize each prompt token to an embedding drawn from the model’s vocabulary.
>   - initialize the prompt with embeddings that enumerate the output classes, similar to the “verbalizers” of Schick and Schütze (2021).
>
> - prefix length: aim to find a minimal length that still performs well. When examining longer prompts (e.g. size 100), we often find several prompt tokens with the same nearest neighbors. This suggests there is either excess capacity in the prompt, or that the **lack of sequential structure** in the prompt representation makes it difficult for the model to localize information to a specific position
>
> - pre-training objective: 这块跟 T5的关系比较大，每太看懂
>
> - 这篇论文数据集用的superglue，但把它们转成了text-to-text 的任务。Each of our prompts train on a single Super- GLUE task; there was no multi-task setup or mixing of training data across tasks. We translate each SuperGLUE dataset into a text-to-text format following Raffel et al. (2020), except that we omit the task names prepended to inputs indicating which SuperGLUE task an example belongs to.
>
>   ![image-20220720102528013](.\working.assets\image-20220720102528013.png)
>
> - domin shift
>
>   ![image-20220720103155295](.\working.assets\image-20220720103155295.png)
>
> - ensemble: Checkpoints are selected via early stopping on the development set, where the stopping metric is the default metric for the dataset, or the average of metrics for datasets evaluated with multiple metrics.
>
>   ![image-20220720103219688](.\working.assets\image-20220720103219688.png)
>
> - interpretability
>
>   - We observe that for a given learned prompt token, the top-5 nearest neighbors form tight semantic clusters. For example, we see lexically similar clusters such as { Technology / technology / Technologies / technological / technologies }, as well as more diverse but still strongly related clusters such as { entirely / completely / totally / altogether / 100% }.
>   - When initializing the prompts using the “classlabel” strategy, we often find that the **class labels persist through training**. Specifically, if a prompt token is initialized to a given label, that label is often among the learned token’s nearest neighbors after tuning. When initializing with the “Random Uniform” or “Sampled Vocab” methods, the class labels can also be found in the nearest neighbors of the prompts; however they tend to appear as neighbors to multiple prompt tokens. This suggests that the model is learning to store the expected output classes in the prompts as reference, and initializing the prompt to outputs classes makes this easier and more centralized.
>   - 
>
> - 出发点
>
>   - prefix token 初始化
>   - prompt tuning 跟pre-train 节点一些设置的关系(比如pre-training objective)
>   - 应用：domain shift
>   - prompt ensemble
>   - interpretability
>   - prefix token 经常重复，可能是因为缺少序列信息



![image-20220720102756330](.\working.assets\image-20220720102756330.png)





## 【2022 openPrompt】

> Title: [OpenPrompt: An Open-source Framework for Prompt-learning](https://aclanthology.org/2022.acl-demo.10.pdf)
>
> author: Ning Ding1, Shengding Hu1, Weilin Zhao1, Yulin Chen5 , Zhiyuan Liu, Hai-Tao Zheng5, Maosong Sun
>
> code：https://github.com/thunlp/OpenPrompt
>
> main：设计了一套prompt 的架构，PLM 主要基于hugging face，自己的主要接口包括：template，vebalizer
>
> ![image-20220731151839111](.\working.assets\image-20220731151839111.png)
>
> ![image-20220731154405328](.\working.assets\image-20220731154405328.png)
>
> points：
>
> - templating strategy
>
>   - **hard(manual) v.s. soft v.s. mix**： manually written template (Schick and Schutze ¨ , 2021) and pure soft template (Lester et al., 2021), Gu et al. (2021) report a mix of manual template tokens and soft (trainable) tokens sometimes yields better results than separate manual template and soft template. In Liu et al. (2021b), a promising performance is achieved by fixing the majority of manual tokens while tuning a small number of the others.
>
>     In Han et al. (2021b), the template is contextualized, which needs to be filled with the head entity and the tail entity to form a complete one, moreover, the output of multiple positions is used in the loss calculation in their template
>
>     Logan IV et al. (2021) design null template with simple concatenation of the inputs and an appended  token.
>
>   - initializing strategy
>
>   - template laguage
>
>     ![image-20220731153108747](D:\workplace\markdown\lab\mytopics\02prompt\working.assets\image-20220731153108747.png)
>
> - PLMs: 三类
>
>   - MLM: BERT (Devlin et al., 2019), RoBERTa (Liu et al., 2019), etc
>   - LM: GPT-3 (Brown et al., 2020) i
>   - Seq2Seq：T5 (Raffel et al., 2020), MASS (Song et al., 2019) and BART (Lewis et al., 2020),
>
> - Verbalizer
>
>   - ManualVerbalizer
>
>     ![image-20220731153313035](.\working.assets\image-20220731153313035.png)
>
>   - AutomaticVerbalizer: Additional to manually-defined verbalizers, we implement automatic verbalizers like AutomaticVerbalizer and KnowledgeableVerbalizer (Hu et al., 2021). Moreover, important operations like calibrations (Zhao et al., 2021) are also realized in OpenPrompt.
>
>   - GenerationVerbalizer: Prompt-learning could also facilitate the unification of NLP tasks. In such kind of paradigm, a span of text (i.e., the target text) is expected to be generated in the masked position. Then the final prediction will be based on a mapping from the target texts to the labels (Ye et al., 2021; Du et al., 2021). To fully support such a paradigm, we implement a novel GenerationVerbalizer,
>
>     ![image-20220731153638755](.\working.assets\image-20220731153638755.png)
>
> - training
>
> - Evaluation
>
>   the evaluation tasks include WebNLG (Gardent et al., 2017) for conditional generation, GLUE (Wang et al., 2018) and SuperGLUE (Wang et al., 2019) for natural language understanding; SemEval (Hendrickx et al., 2010), Few-NERD (Ding et al., 2021b) for information extraction; MNLI (Williams et al., 2017), AG’s News (Zhang et al., 2015), DBPedia (Lehmann et al., 2015) and IMDB (Maas et al., 2011) for text classification; LAMA (Petroni et al., 2019) for knowledge probing.  results  https://github.com/thunlp/OpenPrompt/tree/main/results/
>
> - discussion
>
>   - prompt-learning intuitively **bridges the gap between pre-training and model tuning**. Practically, this paradigm is surprisingly effective in low-data regime (Le Scao and Rush, 2021; Gao et al., 2021). For example, with appropriate template, zero-shot prompt-learning could even outperform 32-shot fine-tuning (Ding et al., 2021a)
>   - Another promising empirical attribute of prompt-learning is the potential to stimulate large-scale PLMs. When it comes to a 10B model, solely optimizing prompts (the parameters of the model are fixed) could achieve comparable performance to full parameter finetuning (Lester et al., 2021). These practical studies imply that we may **use prompts to more effectively and efficiently dig the knowledge kept in PLMs**, leading to a deeper understanding of the underlying principles of their mechanisms (Wei et al., 2021; Qin et al., 2021; Vu et al., 2021).
>
>   - other techniques exploring the parameter-efficient stimulation of large-scale PLMs (Houlsby et al., 2019; Hu et al., 2022; He et al., 2022; Ding et al., 2022).
>   - prompt-based adversarial attacking



## 2021 p-tuning

> title: GPT Understands, Too
>
> author: Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, Jie Tang
>
> institute: Tsinghua
>
> code: https://github.com/THUDM/P-tuning
>
> main: 核心是把离散的promt 变成连续的，也就是实现下图中从公式(1) 到(2) 的转换。但另一方面，存在两个挑战去优化连续的promt：1) **Discreteness**: the original word embedding e ofMhas already become highly discrete after pre-training. If h is initialized with random distribution and then optimized with stochastic gradient descent (SGD), which has been proved to only change the parameters in a small neighborhood (Allen- Zhu et al., 2019), the optimizer would easily fall into local minima. 2) **Association**: another concern would be, intuitively, we believe the values of prompt embeddings hi should be dependent on each other rather than independent. We need some mechanism to associate prompt embeddings with each other.  因此作者额外增加了一个 MLP 和 LSTM 组成的网络对pompt 做一个转化，如图2所示
>
> ![image-20220802221907372](.\working.assets\image-20220802221907372.png)
>
> ![image-20220802222135759](.\working.assets\image-20220802222135759.png)
>
> points：
>
> - PLMs: MLM(BERT, RoBERTa)， LM(GPT, MegatronLM) 
> - MLP 和 LSTM 的计算在inference 的时候可以省略，因为可以提前算好
> - anchor: 作者发现加入一些anchor 会有效果：Besides, we also find that adding few anchor tokens helps some NLU tasks in the SuperGLUE benchmark. For instance, for RTE task, the token “?” within prompt template “[PRE][prompt tokens][HYP]?[prompt tokens][MASK]” is specially added as an anchor token and affects the performance a lot. Usually such anchor words characterize each component, where in this case “?” indicate that “[HYP]” acts as an interrogation part
> - datasets：we conduct extensive experiments on two widely acknowledged natural language understanding benchmarks: LAMA (Petroni et al., 2019) knowledge probing and SuperGlue (Wang et al., 2019b).



# robust

## [2022] robust survey

> Title: [Measure and Improve Robustness in NLP Models: A Survey](https://arxiv.org/pdf/2112.08313.pdf) 
>
> Author: Xuezhi Wang, Haohan Wang, Diyi Yang
>
> Institute: Google, CMU
>
> main: In this paper, we aim to provide a unifying survey of how to **define, measure and improve robustness in NLP**. We first connect multiple definitions of robustness, then unify various lines of work on identifying robustness failures and evaluating models’ robustness. Correspondingly, we present mitigation strategies that are **data-driven**, **model-driven**, and **inductive-prior-based**, with a more systematic view of how to effectively  improve robustness in NLP modelsimprove robustness in NLP models.

**failures in NLP**

- NLP models are still fragile and brittle to out-of-domain data (Hendrycks et al., 2020a; Wang et al., 2019d), 
- adversarial attacks (McCoy et al., 2019; Jia and Liang, 2017; Jin et al., 2020), 
- small perturbation to the input (Ebrahimi et al., 2018; Belinkov and Bisk, 2018).

**Definitions of Robustness in NLP**

- *defenitions*: denote the input as $x$, and its associated gold label for the main task as $y$, assume a model f is trained on $(x, y) ∼ D$ and its prediction over $x$ as $f(x)$; now given test data $(x' , y' ) ∼ D' \neq D$

- *Robustness against Adversarial Attacks* (**synthetic** distribution shift 包括一些基本的数据增强)

  - related: pioneered by (Szegedy et al., 2013; Goodfellow et al., 2015), and later extended to NLP, such as (Ebrahimi et al., 2018; Alzantot et al., 2018; Li et al., 2019; Feng et al., 2018; Kuleshov et al., 2018; Jia et al., 2019; Zang et al., 2020; Pruthi et al., 2019; Wang et al., 2019e; Garg and Ramakrishnan, 2020; Tan et al., 2020a,b; Schwinn et al., 2021; Li et al., 2021; Boucher et al., 2022) and multilingual adversaries (Yang et al., 2019; Tan and Joty, 2021).

  - NLP 的一些特殊性让其可以更容易被 Adversarial Attack：human’s remarkable ability in understanding a large set of synonyms (Li et al., 2020) or interesting characteristics in ignoring the exact order of letters (Wang et al., 2020b) are often opportunities to create adversarial examples

  - *NLP models’ **vulnerability** against attacks:*  A related line of work such as data-poisoning (Wallace et al., 2021) and weight-poisoning (Kurita et al., 2020) exposes NLP models’ vulnerability against attacks during the training process. One can refer to more comprehensive reviews and broader discussions on this topic in Zhang et al. (2020c) and Morris et al. (2020b).

  - Assumptions around **Label-preserving**

    Wang et al. (2021b) studied several existing text perturbation techniques and found that a **significant portion of perturbed examples are not label-preserving** (despite their label-preserving assumptions), or the resulting labels have a high disagreement among human raters (i.e., can even fool humans). Morris et al. (2020a) also call for more attention to the validity of perturbed examples for a more accurate robustness evaluation

    work in NLP **follows label-preserving assumption** with small text perturbations like token and character swapping :  g (Alzantot et al., 2018; Jin et al., 2020; Ren et al., 2019; Ebrahimi et al., 2018), paraphrasing (Iyyer et al., 2018; Gan and Ng, 2019), semantically equivalent adversarial rules (Ribeiro et al., 2018), and adding distractors (Jia and Liang, 2017).

    without the label-preserving assumption:  (Gardner et al., 2020; Kaushik et al., 2019; Schlegel et al., 2021). $y' \neq y$

  - Assumptions around and **Semantic-preserving**

    One alternative notion is whether the perturbation from x to x 0 is “semantic-preseving” (Alzantot et al., 2018; Jin et al., 2020; Ren et al., 2019) or “semantic-modifying” (Shi and Huang, 2020; Jia and Liang, 2017). Note this is slightly different from the above label-preserving assumptions, as it is defined over the perturbations on $(x, x' )$ rather than making an assumption on $(y, y' )$, e.g., semantic-modifying perturbations can be either label-preserving (Jia and Liang, 2017; Shi and Huang, 2020) or label-changing (Gardner et al., 2020; Kaushik et al., 2019).

    ![image-20220819101823227](.\working.assets\image-20220819101823227.png)

- *Robustness under Distribution Shift* (**natural** distribution shift)

  - concept: Another line of research focuses on $(x' , y')$ drawn from a different distribution that is naturally occurring (Hendrycks et al., 2021), where robustness can be defined around model’s performance under distribution shift. **Different from work on domain adaptation** (Patel et al., 2015; Wilson and Cook, 2020) **and transfer learning** (Pan and Yang, 2010), **existing definitions of robustness are closer to the concept of domain generalization** (Muandet et al., 2013; Gulrajani and Lopez-Paz, 2021), **or out-of-distribution generalization** to unforeseen distribution shifts (Hendrycks et al., 2020a), where the test data (either labeled or unlabeled) is assumed not available during training,

    In the context of NLP, robustness to natural distribution shifts can also mean models’ performance should not degrade due to the differences in grammar errors, dialects, speakers,languages (Craig and Washington, 2002; Blodgett et al., 2016; Demszky et al., 2021), or newly collected datasets for the same task but in different domains (Miller et al., 2020).

  - related

    Another closely connected line of research is **fairness**, which has been studied in various NLP applications, see (Sun et al., 2019) for a more in-depth survey in this area. For example, gendered stereotypes or biases have been observed in NLP tasks including co-reference resolution (Zhao et al., 2018a; Rudinger et al., 2017), occupation classification (De-Arteaga et al., 2019), and neural machine translation (Prates et al., 2019; Font and Costa-jussà, 2019).

    **dataset biases**：The robustness failures can sometimes be attributed to dataset biases, i.e., biases introduced during dataset collection (Fouhey et al., 2018) or human annotation artifacts (Gururangan et al., 2018; Geva et al., 2019; Rudinger et al., 2017), which could affect how well a model trained from this dataset generalizes, and how accurately we estimate a model’s performance. For example, Lewis et al. (2021) show there is a significant test-train data overlap in a set of opend omain question-answering benchmarks, and many QA models perform substantially worse on questions that cannot be memorized from training data. In natural language inference, McCoy et al. (2019) show that commonly used crowdsourced datasets for training NLI models might make certain syntactic heuristics more easily adopted by statistical learners. Further, Bras et al. (2020) propose to use a lightweight adversarial filtering approach to filter dataset biases, which is approximated using each instance’s predictability score.

- transferability of the two categories

  In the vision domain, Taori et al. (2020) investigate models’ robustness to natural distribution shift, and show that robustness to synthetic distribution shift might offer little to no robustness improvement under natural distribution shift. Some studies show NLP models might not generalize to unseen adversarial patterns (Huang et al., 2020; Jha et al., 2020; Joshi and He, 2021), but more work is needed to systematically bridge the gap between NLP models’ robustness under natural and synthetic distribution shifts

- Further, in certain applications, model “robustness” can also be connected with **models’ instability** (Milani Fard et al., 2016), or models having **poorly-calibrated uncertainty estimation** (Guo et al., 2017), where Bayesian methods (Graves, 2011; Blundell et al., 2015), dropout-based (Gal and Ghahramani, 2016; Kingma et al., 2015) and ensemble-based approaches (Lakshminarayanan et al., 2017) have been proposed to improve models’ uncertainty estimation. Recently, Ovadia et al. (2019) have shown models’ uncertainty estimation can degrade significantly under distributional shift, and **call for more work to ensure a model “knows when it doesn’t know” by giving lower uncertainty estimates over out-of-distribution data.** This is another example where models can be less robust under distributional shifts, and again emphasizes the need of building more unified benchmarks to measure a model’s performance (e.g., robust accuracy, calibration, stability) under distribution shifts, in addition to in-distribution accuracy.



**identify measure a model’s robustness**

- *tasks like text classification and sequence labeling*: we can measure a model’s robustness by its performance on $D'$ , e.g., using the model’s robust accuracy (Tsipras et al., 2019; Yang et al., 2020), defined as $E_{(x' ,y')∼D'}[f(x' ) = y']$

- *tasks like text generation:* robustness is less well defined and can manifest as **positional bias** (Jung et al., 2019; Kryscinski et al., 2019), or **hallucination** (Maynez et al., 2020; Parikh et al., 2020; Zhou et al., 2021). One major challenge here is a lack of robust metrics in evaluating the quality of the generated text (Sellam et al., 2020; Zhang et al., 2020b), i.e., we need a reliable metric to determine the relationship between $f(x')$ and $y'$ when both are open-ended texts.

- **identify robustness failures**: The identified robustness failure patterns are usually organized into challenging/adversarial benchmark datasets to more accurately measure an NLP model’s robustness.

  understanding and measuring robustness in NLP models (Tu et al., 2020; Sagawa et al., 2020b; Geirhos et al., 2020) across various NLP tasks

  categories: human priors, error analyses, model-based

- human priors and error analyses driven

  - Natural Language Inference

    Gururangan et al. (2018) found that current NLI models are likely to identify the label by relying only on the hypothesis, and Poliak et al. (2018) provided similar augments that using a hypothesis-only model can outperform a set of strong baselines.

  - Question Answering

    Jia and Liang (2017) proposed to generate adversarial QA examples by concatenating an adversarial distracting sentence at the end of a paragraph. Miller et al. (2020) built four new test sets for the Stanford Question Answering Dataset (SQuAD) and found most question answering systems fail to generalize to this new data, calling for new evaluation metrics towards natural distribution shifts.

  - Machine Translation

    Belinkov and Bisk (2018) found that character-based neural machine translation (NMT) models are brittle under noisy data, where noises (e.g., typos, misspellings, etc) are synthetically generated using possible lexical replacements. Data augmentation with **artificially introduced grammatical errors** (Anastasopoulos et al., 2019) or with **random synthetic noises** (Vaibhav et al., 2019; Karpukhin et al., 2019) can make the system more robust to such spurious patterns. On the other hand, Wang et al. (2020b) showed another approach by limiting the input space of the characters so that the models will be likely to perceive data typos and misspellings.

  - Syntactic and Semantic Parsing: 

    Robust parsing has been studied in several existing works (Lee et al., 1995; Aït-Mokhtar et al., 2002). More recent work showed that neural semantic parsers are still not robust against lexical and stylistic variations, or meaning-preserving perturbations (Marzinotto et al., 2019; Huang et al., 2021), and proposed ways to improve their robustness through data augmentation (Huang et al., 2021) and adversarial learning (Marzinotto et al., 2019).

  - Text Generation Existing work found that text generation models also suffer from robustness issues, e.g., text summarization models suffer from positional bias (Jung et al., 2019), layout bias (Kryscinski et al., 2019), and a lack of faithfulness and factuality (Kryscinski et al., 2019; Maynez et al., 2020; Chen et al., 2021b); data-to-text models sometimes hallucinate texts that are not supported by the data (Parikh et al., 2020; Wang et al., 2020d). In addition, Sellam et al. (2020); Zhang et al. (2020b) pointed out the deficiency of existing automatic evaluation metrics and proposed new metrics to better align the generation quality with human judgements.

- model based identification

  - identify robustness failures that are task-agnostic like **white-box text attack** methods (Ebrahimi et al., 2018; Alzantot et al., 2018; Jin et al., 2020), and even input-agnostic like universal adversarial triggers (Wallace et al., 2019a) and natural attack triggers (Song et al., 2021).

  - **learn an additional model to capture biases**, e.g., in visual question answering, Clark et al. (2019) train a naive model to predict prototypical answers based on the question only irrespective of the context; He et al. (2019); Utama et al. (2020a) propose to learn a biased model that only uses dataset-bias related features. This framework has also been used to capture unknown biases assuming that the lower capacity model learns to capture relatively shallow correlations during training (Clark et al., 2020). In addition, Wang and Culotta (2020a) identify model shortcuts by training classifiers to better distinguish “spurious” correlations from “genuine” ones based on human annotated examples.

  - Model-in-the-loop vs. Human-in-the-loop

    adopts human-in-the-loop to generate challenging examples: Counterfacutal-NLI (Kaushik et al., 2019) and Natural-Perturbed-QA (Khashabi et al., 2020). 

    applies modelin-the-loop(might also introduce biases towards the particular model used):  SWAG (Zellers et al., 2018) was introduced that fooled most models at the time of publishing but was soon “solved” after BERT (Devlin et al., 2019) was introduced.  As a result, Yuan et al. (2021) present a study over the transferability of adversarial examples, and Contrast Sets (Gardner et al., 2020) intentionally avoid using model-in-the-loop. 

    adopts adversarial humanand-model-in-the-loop to create more difficult examples:  Adv-QA (Bartolo et al., 2020), Adv-Quizbowl (Wallace et al., 2019b), ANLI (Nie et al., 2020), and Dynabench (Kiela et al., 2021).

- benchmarks

  - Natural Language Inference

    Naik et al. (2018) sampled misclassified examples and analyzed their potential sources of errors, which are then grouped into a typology of common reasons for error. Such error types then served as the bases to construct the **stress test set**, to further evaluate whether NLI models have the ability to make real inferential decisions, or simply rely on sophisticated pattern matching.  Kaushik et al. (2019) asked humans to generate **counterfactual NLI** examples, to better understand what features are causal and encourage models to learn those features.

![image-20220819103427564](.\working.assets\image-20220819103427564.png)

**why models exhibit a lack of robustness**

- **spurious features:** spurious features are commonly defined as features that do not causally affect a task’s label (Srivastava et al., 2020; Wang and Culotta, 2020b): they correlate with task labels but fail to transfer to more challenging test conditions or out-of-distribution data (Geirhos et al., 2020). Some other work defined it as “prediction rules that work for the majority examples but do not hold in general” (Tu et al., 2020). 

  Such spurious correlations are sometimes referred as dataset bias (Clark et al., 2019; He et al., 2019), annotation artifacts (Gururangan et al., 2018), or group shift (Oren et al., 2019) in the literature.

  Theoretical discussions connecting these fields have also been offered by crediting a reason of model’s lack of robustness in either distribution shift or adversarial attack to model’s learning of spurious features (Wang et al., 2021c).

  Further, evidence showed that controlling model’s learning in spurious features will improve model’s performances in distribution shifts (Wang et al., 2019a,b); also, discussions on the connections between adversarial robustness and learning of spurious features have been raised (Ilyas et al., 2019; Wang et al., 2020a)

  One interesting phenomenon observed by (Liu et al., 2019) is to attribute models’ robustness failures to blind spots in the training data, or the intrinsic learning ability of the model. The authors found that both patterns are possible: in some cases models can be inoculated via being exposed to a small amount of challenging data, similar to the data augmentation approaches

  other hand, some challenging patterns remain difficult which connects to the larger question around generalizability to unseen adversarial and counterfactual patterns (Huang et al., 2020; Jha et al., 2020; Joshi and He, 2021), which is relatively under-explored but deserves much attention.



**Improve Model Robustness**

- data-driven: see paper

- model-based  and training-scheme-based

  - pre-training

    Recent work has demonstrated pretraining as an effective way to improve NLP models’ out-of-distribution robustness (Hendrycks et al., 2020a; Tu et al., 2020)

     Tu et al. (2020) showed a few other factors can also contribute to robust accuracy, including larger model size, more fine-tuning data, and longer fine-tuning. A similar observation is made by Taori et al. (2020) in the vision domain

  - Better Use of Minority Examples

    Further, there are several works that propose to robustify the models via a better use of minority examples, e.g., examples that are under-represented in the training distribution, or examples that are harder to learn. For example, Yaghoobzadeh et al. (2021) proposed to first fine-tune the model on the full data, and then on minority examples only.

    DRO: e training strategy with an emphasis on a subset of samples that are particularly hard for the model to learn is sometimes also referred to as group DRO (Sagawa et al., 2020a), as an extension of vanilla **distributional robust optimization (DRO)** (Ben-Tal et al., 2013; Duchi et al., 2021). Extensions of DRO are mostly discussing the strategies on how to identify the samples considered as minority: Nam et al. (2020) trained two models in parallel, where the “debiased” model focuses on examples not learned by the “biased” model; Lahoti et al. (2020) used an adversary model to identify samples that are challenging to the main model; Liu et al. (2021) proposed to train the model a second time via up-weighting examples that have high training losses during the first time.

- inductive-prior-based  : Another thread is to introduce inductive bias (i.e., to regularize the hypothesis space) to force the model to discard some spurious features.  To achieve this goal, one usually needs to first construct a side component to inform the main model about the misaligned features, and then to regularize the main model according to the side component. The construction of this side component usually relies on prior knowledge of what the misaligned features are. 

  Then, methods can be built accordingly to counter the features such as label-associated keywords (He et al., 2019), label-associated text fragments (Mahabadi et al., 2020), and general easy-to-learn patterns of data (Nam et al., 2020).

  Similarly, Clark et al. (2019, 2020); Utama et al. (2020a,b) propose to **ensemble** with a model explicitly capturing bias, where the main model is trained together with this “bias-only” model such that the main model is discouraged from using biases. More recent work (Xiong et al., 2021) shows the ensemble-based approaches can be further improved via better calibrating the bias-only model.

  additional **regularizers** have been introduced for robust fine-tuning over pre-trained models, e.g., mutual-information-based regularizers (Wang et al., 2021a) and smoothness-inducing adversarial regularization (Jiang et al., 2020)

  domain adversarial neural network (Ganin et al., 2016). This line of work also inspires a family of methods forcing the model to learn auxiliary-annotationinvariant representations with a side component (Ghifary et al., 2016; Wang et al., 2017; Rozantsev et al., 2018; Motiian et al., 2017; Li et al., 2018; Wang et al., 2019c; Vernikos et al., 2020)

  (the above is mainly training for small empirical loss across different domains or distributions in addition to forcing the model to **be invariant to domain-specific spurious features**)

  **invariant risk minimization (IRM)** (Arjovsky et al., 2019) introduces the idea of invariant predictors across multiple environments, which was later followed and discussed by a variety of extensions (Choe et al., 2020; Ahmed et al., 2020; Rosenfeld et al., 2021). More recently, Dranker et al. (2021) applied IRM in natural language inference and found that a more naturalistic characterization of the problem setup is needed.

- causal intervention

  Casual analyses have also been utilized to examine robustness. Srivastava et al. (2020) leverage humans’ common sense knowledge of causality to augment training examples with a potential unmeasured variable, and propose a DRO-based approach to encourage the model to be robust to distribution shifts over the unmeasured variables. Balashankar et al. (2021) study the effect of secondary attributes, or confounders, and propose context-aware counterfactuals that take into account the impact of secondary attributes to improve models’ robustness. Veitch et al. (2021) propose to learn approximately counterfactual invariant predictors dependent on causal structures of the data, and show it can help mitigate spurious correlations in text classification

- (Interestingly, statistical work has shown that many of these mitigation methods are optimizing the same robust machine learning generalization error bound (Wang et al., 2021c).)

**Robustness in Vision vs. in NLP**

- Continuous vs. Discrete in Search Space:  his particularly posed a challenge towards the adversarial attack and defense regime when the study in vision is transferred to NLP (Lei et al., 2019; Zhang et al., 2020c), in the sense that simple gradient-based adversarial attacks will not directly translate to meaningful attacks in the discrete text space
- Perceptible to Human vs. Not: NLP attack 更容易被感知，但从阅读的角度来看，通常不影响整体阅读。 Instead of being imperceptible, the adversarial attacks in NLP typically are bounded by the fact that the meaning of the sentences are not altered (despite being perceptible). On the other hand, there are ways to generate samples where the changes, although being perceptible, are often ignored by human brain due to some psychological prior on how a human processes the text (Anastasopoulos et al., 2019; Wang et al., 2020b)
- domain adaptation：Vision 在adaptions 的时候 数据分布范围是一致的(像素都是0-255)，但NLP 的词汇可能不一样甚至不同语言。domain adaptation of NLP sometimes studies the regime where the supports of the data differ, e.g., the vocabularies can be significantly different in cross-lingual studies (Abad et al., 2020; Zhang et al., 2020a).



 **Open Questions**

- Identifying Unknown Robustness Failures

  Existing identification around robustness failures rely heavily on human priors and error analyses, which usually pre-define a small or limited set of patterns that the model could be vulnerable to. This requires extensive amount of expertise and efforts, and might still suffer from human or subjective biases in the end. 

- Interpreting and Mitigating Spurious Correlations

  recent work (Wallace et al., 2019c; Wang et al., 2021d; Zhang et al., 2021) show **interpretability** methods can be utilized to better understand how a model makes its decision, which in turn can be used to uncover models’ bias, diagnose errors, and discover spurious correlations.

  Furthermore, the mitigation of spurious correlations often suffers from the trade-off between removing shortcuts and sacrificing model performance (Yang et al., 2020; Zhang et al., 2019a)

- Unified Framework to Evaluate Robustness

  With a variety of potential spurious patterns in NLP models, it becomes increasingly challenging for developers and practitioners to quickly evaluate the robustness and quality of their models. 

  This calls for more unified benchmarking efforts such as **CheckList (Ribeiro et al., 2020), Reliability Testing (Tan et al., 2021), Robustness Gym (Goel et al., 2021) and Dynabench (Kiela et al., 2021)**, to facilitate fast and easy evaluation of robustness.

- User Centered Measures and Mitigation

  Based on the dualprocess models of information processing, humans use two different processing styles (Evans, 2010). One is a quick and automatic style that relies on well-learned information and heuristic cues. The other is a qualitatively different style that is slower, more deliberative, and requires more reflective reasoning

- Connections between Human-like Linguistic Generalization and NLP Generalization

  Linzen (2020) argue NLP models should behave more like humans to achieve better generalization consistently



## [2020] domain adaption survey

> Title: [Neural Unsupervised Domain Adaptation in NLP—A Survey](https://aclanthology.org/2020.coling-main.603.pdf) (2020 ACL)
>
> Author: Alan Ramponi, Barbara Plank
>
> Institute: University of Trento, Italy, Microsoft
>
> repo: https://github.com/bplank/awesome-neural-adaptation-in-NLP
>
> ###### idea: dataset shift -> model shift