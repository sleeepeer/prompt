# prompt

10.17-10.30的调研内容

*此为论文列表，详细内容在子目录*

**所有论文都已经上传了pdf文件，后面也标了链接*



## survey1 低层general，高层specific

* **AutoLR: Layer-wise Pruning and Auto-tuning of Learning Rates in Fine-tuning of Deep Networks**	[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/16350)

  之前你给我那篇，里面提了浅/深层general/specific的问题，引出了以下两篇讨论这个的文章 10.18

* **How transferable are features in deep neural networks?**	[pdf](https://proceedings.neurips.cc/paper/2014/hash/375c71349b295fbe2dcdca9206f20a06-Abstract.html)

  量化了general和specific，讨论了转移过程（在哪些层） ，也有固定浅层feature的做法 10.18

* **Visualizing and Understanding Convolutional Networks**	[pdf](https://link.springer.com/chapter/10.1007/978-3-319-10590-1_53)

​		可视化CNN各层学习到了哪些feature，实现上提出DCNN反卷积网络，对解释general/specific有帮助 10.19

* **From Generic to Specific Deep Representations for Visual Recognition**  [pdf](https://www.cv-foundation.org/openaccess/content_cvpr_workshops_2015/W03/html/Azizpour_From_Generic_to_2015_CVPR_paper.html)



## survey2 类似cache feature的工作（transfer learning + prompt）

* **How transferable are features in deep neural networks?（survey1）**	[pdf](https://proceedings.neurips.cc/paper/2014/hash/375c71349b295fbe2dcdca9206f20a06-Abstract.html)

* **Convolutional Neural Networks for Medical Image Analysis: Full Training or Fine Tuning?** 	[pdf](https://ieeexplore.ieee.org/abstract/document/7426826)
	提出incremental fine-tuning，从最后层开始依次递增做fine-tune
	
* **迁移学习transfer learning博客** https://www.zhihu.com/question/41979241
  列了几篇DNN迁移学习，domain adaptaion论文
  
* **SPoT: Better Frozen Model Adaptation through Soft Prompt Transfer**  [pdf](https://arxiv.org/pdf/2110.07904.pdf) 
  [【2021.10.17】SPoT：非常简单的prompt预训练方式+很好的task transfer效果 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/422472763)
  通过上游任务学习到的prompt来初始化下游的任务prompt；量化了任务间的相似度

* **迁移学习transfer learning博客** https://www.zhihu.com/question/41979241

  列了几篇DNN迁移学习，domain adaptaion论文

---

  ### 待考察

  * https://arxiv.org/abs/2210.00990
  * https://arxiv.org/abs/2110.07904
  * 小综述 [Prompt Tuning 近期研究进展 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/422921903)



## other

* prompt概述与论文资源 [近代自然语言处理技术发展的“第四范式” - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/395115779)
* [pfliu-nlp/NLPedia-Pretrain (github.com)](https://github.com/pfliu-nlp/NLPedia-Pretrain)
* prompt鲁棒性研究 [2110.07280.pdf (arxiv.org)](https://arxiv.org/pdf/2110.07280.pdf)

* colab教程[Colab使用教程（超级详细版）及Colab Pro/Pro+评测 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/527663163)
