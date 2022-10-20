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

## survey2 类似cache feature的工作

* **（survey1）How transferable are features in deep neural networks?**	[pdf](https://proceedings.neurips.cc/paper/2014/hash/375c71349b295fbe2dcdca9206f20a06-Abstract.html)

* **迁移学习transfer learning博客** https://www.zhihu.com/question/41979241、

	列了几篇DNN迁移学习好文，domain adaptaion好文

## other

* **Convolutional Neural Networks for Medical Image Analysis: Full Training or Fine Tuning?** 	[pdf](https://ieeexplore.ieee.org/abstract/document/7426826)

​		Partial-tuning只对深层做fine-tune

* 
